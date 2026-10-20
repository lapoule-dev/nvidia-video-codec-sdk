//! A safe decoder: an H.264 or HEVC elementary stream in, pictures out, in
//! the order they are shown, copied to host memory.
//!
//! See [NVIDIA Video Codec SDK - Video Decoder API Programming Guide](https://docs.nvidia.com/video-technologies/video-codec-sdk/12.0/nvdec-video-decoder-api-prog-guide/index.html).
//!
//! The stream is handed in Annex B (start codes), one access unit at a
//! time; NVIDIA's own parser (`cuvidCreateVideoParser`) reads it, so no
//! bitstream parsing happens here. The parser calls back into this module
//! on the thread that handed it the data: once the stream says its format
//! (the decoder is made then), for each picture to decode, and for each
//! picture to show. A picture shown is mapped, copied to the host and
//! unmapped there and then.
//!
//! Pictures come out as NV12 (8 bits) or P016 (16-bit samples, the
//! stream's bits in the high ones: P010 for a 10-bit stream), their rows
//! tightly packed: the luma's `height` rows, then `height / 2` rows of
//! interleaved chroma.

use std::{
    ffi::{c_int, c_uint, c_ulong, c_ulonglong, c_void},
    ptr,
    sync::Arc,
};

use cudarc::driver::{
    sys::{self as cuda, CUmemorytype, CUresult, CUDA_MEMCPY2D},
    CudaContext,
};

use crate::sys::{cuviddec::*, nvcuvid::*};

/// What a [`Decoder`] decodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeCodec {
    /// H.264 / AVC.
    H264,
    /// HEVC / H.265, 8 or 10 bits.
    Hevc,
}

impl DecodeCodec {
    fn cuda(self) -> cudaVideoCodec {
        match self {
            DecodeCodec::H264 => cudaVideoCodec::cudaVideoCodec_H264,
            DecodeCodec::Hevc => cudaVideoCodec::cudaVideoCodec_HEVC,
        }
    }
}

/// Why decoding failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeError(pub String);

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NVDEC: {}", self.0)
    }
}

impl std::error::Error for DecodeError {}

fn check(result: CUresult, what: &str) -> Result<(), DecodeError> {
    if result == CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(DecodeError(format!("{what}: {result:?}")))
    }
}

/// One picture, in the order it is shown.
#[derive(Clone, PartialEq, Eq)]
pub struct Picture {
    /// The displayed width and height, in pixels.
    pub width: u32,
    pub height: u32,
    /// The stream's bits per sample: 8 (NV12) or more (P016).
    pub bit_depth: u8,
    /// The timestamp handed in with the access unit it came from.
    pub timestamp: i64,
    /// NV12 or P016, rows tightly packed (see the module).
    pub data: Vec<u8>,
}

impl std::fmt::Debug for Picture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Picture")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("bit_depth", &self.bit_depth)
            .field("timestamp", &self.timestamp)
            .field("bytes", &self.data.len())
            .finish()
    }
}

impl Picture {
    /// Bytes a sample takes: 1 for NV12, 2 for P016.
    #[must_use]
    pub fn bytes_per_sample(&self) -> usize {
        if self.bit_depth > 8 {
            2
        } else {
            1
        }
    }
}

/// What the callbacks share with the decoder: boxed, so the parser's
/// pointer to it never moves.
struct State {
    codec: DecodeCodec,
    lock: CUvideoctxlock,
    decoder: CUvideodecoder,
    /// The decoder's displayed size, bit depth and surface height.
    format: Option<Format>,
    shown: Vec<Picture>,
    failure: Option<DecodeError>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Format {
    width: u32,
    height: u32,
    bit_depth: u8,
}

/// An NVDEC decoder and its parser, on one CUDA context.
pub struct Decoder {
    ctx: Arc<CudaContext>,
    parser: CUvideoparser,
    state: Box<State>,
}

impl std::fmt::Debug for Decoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Decoder")
            .field("codec", &self.state.codec)
            .finish_non_exhaustive()
    }
}

impl Decoder {
    /// A decoder of `codec` on `ctx`.
    ///
    /// # Errors
    ///
    /// When the context cannot be made current, or the parser not made.
    pub fn new(ctx: Arc<CudaContext>, codec: DecodeCodec) -> Result<Self, DecodeError> {
        ctx.bind_to_thread()
            .map_err(|e| DecodeError(format!("CUDA context: {e:?}")))?;
        let mut lock: CUvideoctxlock = ptr::null_mut();
        // SAFETY: the context is alive for as long as `ctx`, which the
        // decoder keeps.
        check(
            unsafe { cuvidCtxLockCreate(&mut lock, ctx.cu_ctx()) },
            "context lock",
        )?;
        let mut state = Box::new(State {
            codec,
            lock,
            decoder: ptr::null_mut(),
            format: None,
            shown: Vec::new(),
            failure: None,
        });
        // SAFETY: a parameter block of plain values, pointers and optional
        // callbacks: all zero is "none of them".
        let mut params: CUVIDPARSERPARAMS = unsafe { std::mem::zeroed() };
        params.CodecType = codec.cuda();
        // Raised by the sequence callback's answer to what the stream needs.
        params.ulMaxNumDecodeSurfaces = 1;
        // One picture of delay: what NVIDIA's own samples use for a decode
        // that is not low-latency.
        params.ulMaxDisplayDelay = 1;
        params.pUserData = ptr::addr_of_mut!(*state).cast();
        params.pfnSequenceCallback = Some(on_sequence);
        params.pfnDecodePicture = Some(on_decode);
        params.pfnDisplayPicture = Some(on_display);
        let mut parser: CUvideoparser = ptr::null_mut();
        // SAFETY: `params` is complete, and `state` outlives the parser
        // (both are dropped together, the parser first).
        let made = unsafe { cuvidCreateVideoParser(&mut parser, &mut params) };
        if let Err(e) = check(made, "parser") {
            // SAFETY: made above, used by nothing.
            unsafe {
                let _ = cuvidCtxLockDestroy(lock);
            }
            return Err(e);
        }
        Ok(Self { ctx, parser, state })
    }

    /// Decodes one access unit, Annex B, handed with `timestamp`; returns
    /// the pictures this made ready to show, in their order.
    ///
    /// # Errors
    ///
    /// When the stream cannot be decoded.
    pub fn decode(&mut self, access_unit: &[u8], timestamp: i64) -> Result<Vec<Picture>, DecodeError> {
        self.parse(access_unit, timestamp, CUvideopacketflags::CUVID_PKT_TIMESTAMP as c_ulong)
    }

    /// The end of the stream: every picture still waiting to be shown.
    ///
    /// # Errors
    ///
    /// When the stream cannot be decoded.
    pub fn flush(&mut self) -> Result<Vec<Picture>, DecodeError> {
        self.parse(&[], 0, CUvideopacketflags::CUVID_PKT_ENDOFSTREAM as c_ulong)
    }

    fn parse(&mut self, data: &[u8], timestamp: i64, flags: c_ulong) -> Result<Vec<Picture>, DecodeError> {
        self.ctx
            .bind_to_thread()
            .map_err(|e| DecodeError(format!("CUDA context: {e:?}")))?;
        let mut packet = CUVIDSOURCEDATAPACKET {
            flags,
            payload_size: data.len() as c_ulong,
            payload: if data.is_empty() { ptr::null() } else { data.as_ptr() },
            timestamp,
        };
        // SAFETY: the packet points at `data`, alive across the call; the
        // callbacks it runs reach `state`, which `self` holds.
        let parsed = unsafe { cuvidParseVideoData(self.parser, &mut packet) };
        if let Some(failure) = self.state.failure.take() {
            return Err(failure);
        }
        check(parsed, "parse")?;
        Ok(std::mem::take(&mut self.state.shown))
    }
}

impl Drop for Decoder {
    fn drop(&mut self) {
        let _ = self.ctx.bind_to_thread();
        // SAFETY: each was made by this decoder and is destroyed once.
        unsafe {
            let _ = cuvidDestroyVideoParser(self.parser);
            if !self.state.decoder.is_null() {
                let _ = cuvidDestroyDecoder(self.state.decoder);
            }
            let _ = cuvidCtxLockDestroy(self.state.lock);
        }
    }
}

/// The parser's user data, as the state it is.
///
/// # Safety
///
/// `user` must be the `State` a [`Decoder`] gave its parser.
unsafe fn state<'a>(user: *mut c_void) -> &'a mut State {
    &mut *user.cast::<State>()
}

/// The stream says its format: the decoder is made for it. Returns the
/// decode surfaces the parser should use, 0 on failure.
unsafe extern "C" fn on_sequence(user: *mut c_void, format: *mut CUVIDEOFORMAT) -> c_int {
    let state = state(user);
    let format = &*format;
    match make_decoder(state, format) {
        Ok(surfaces) => surfaces,
        Err(e) => {
            state.failure = Some(e);
            0
        }
    }
}

unsafe fn make_decoder(state: &mut State, format: &CUVIDEOFORMAT) -> Result<c_int, DecodeError> {
    if format.chroma_format != cudaVideoChromaFormat::cudaVideoChromaFormat_420 {
        return Err(DecodeError(format!("{:?}: only 4:2:0 is decoded here", format.chroma_format)));
    }
    let area = format.display_area;
    let shown = Format {
        width: (area.right - area.left).max(0) as u32,
        height: (area.bottom - area.top).max(0) as u32,
        bit_depth: format.bit_depth_luma_minus8 + 8,
    };
    // What the stream needs, and a few for the parser's display delay.
    let surfaces = c_int::from(format.min_num_decode_surfaces).max(1) + 4;
    if state.format == Some(shown) && !state.decoder.is_null() {
        return Ok(surfaces);
    }
    if !state.decoder.is_null() {
        check(cuvidDestroyDecoder(state.decoder), "destroy decoder")?;
        state.decoder = ptr::null_mut();
    }
    let mut create: CUVIDDECODECREATEINFO = std::mem::zeroed();
    create.ulWidth = c_ulong::from(format.coded_width);
    create.ulHeight = c_ulong::from(format.coded_height);
    create.ulNumDecodeSurfaces = surfaces as c_ulong;
    create.CodecType = state.codec.cuda();
    create.ChromaFormat = format.chroma_format;
    create.ulCreationFlags = cudaVideoCreateFlags::cudaVideoCreate_PreferCUVID as c_ulong;
    create.bitDepthMinus8 = c_ulong::from(format.bit_depth_luma_minus8);
    create.ulMaxWidth = c_ulong::from(format.coded_width);
    create.ulMaxHeight = c_ulong::from(format.coded_height);
    create.display_area.left = area.left as i16;
    create.display_area.top = area.top as i16;
    create.display_area.right = area.right as i16;
    create.display_area.bottom = area.bottom as i16;
    create.OutputFormat = if format.bit_depth_luma_minus8 > 0 {
        cudaVideoSurfaceFormat::cudaVideoSurfaceFormat_P016
    } else {
        cudaVideoSurfaceFormat::cudaVideoSurfaceFormat_NV12
    };
    create.DeinterlaceMode = cudaVideoDeinterlaceMode::cudaVideoDeinterlaceMode_Weave;
    create.ulTargetWidth = c_ulong::from(shown.width);
    create.ulTargetHeight = c_ulong::from(shown.height);
    create.ulNumOutputSurfaces = 2;
    create.vidLock = state.lock;
    let mut decoder: CUvideodecoder = ptr::null_mut();
    check(cuvidCreateDecoder(&mut decoder, &mut create), "create decoder")?;
    state.decoder = decoder;
    state.format = Some(shown);
    Ok(surfaces)
}

/// A picture to decode.
unsafe extern "C" fn on_decode(user: *mut c_void, picture: *mut CUVIDPICPARAMS) -> c_int {
    let state = state(user);
    if state.decoder.is_null() {
        state.failure = Some(DecodeError("a picture before the stream's format".into()));
        return 0;
    }
    match check(cuvidDecodePicture(state.decoder, picture), "decode") {
        Ok(()) => 1,
        Err(e) => {
            state.failure = Some(e);
            0
        }
    }
}

/// A picture to show: mapped, copied to the host, unmapped.
unsafe extern "C" fn on_display(user: *mut c_void, info: *mut CUVIDPARSERDISPINFO) -> c_int {
    let state = state(user);
    if info.is_null() {
        // The end of the stream.
        return 1;
    }
    match copy_out(state, &*info) {
        Ok(picture) => {
            state.shown.push(picture);
            1
        }
        Err(e) => {
            state.failure = Some(e);
            0
        }
    }
}

unsafe fn copy_out(state: &mut State, info: &CUVIDPARSERDISPINFO) -> Result<Picture, DecodeError> {
    let format = state
        .format
        .ok_or_else(|| DecodeError("a picture shown before the stream's format".into()))?;
    let mut process: CUVIDPROCPARAMS = std::mem::zeroed();
    process.progressive_frame = info.progressive_frame;
    process.second_field = info.repeat_first_field + 1;
    process.top_field_first = info.top_field_first;
    process.unpaired_field = c_int::from(info.repeat_first_field < 0);
    let (mut device, mut pitch): (c_ulonglong, c_uint) = (0, 0);
    check(
        cuvidMapVideoFrame64(state.decoder, info.picture_index, &mut device, &mut pitch, &mut process),
        "map",
    )?;
    let bytes = if format.bit_depth > 8 { 2 } else { 1 };
    let (width, height) = (format.width as usize, format.height as usize);
    let row = width * bytes;
    let chroma_rows = height.div_ceil(2);
    let mut data = vec![0u8; row * (height + chroma_rows)];
    // The chroma plane starts after the surface's luma, its height rounded
    // up to even, as NVIDIA's samples read it.
    let surface_height = (height + 1) & !1;
    let copied = (|| {
        for (from, to, rows) in [
            (device, 0usize, height),
            (device + (pitch as u64) * surface_height as u64, row * height, chroma_rows),
        ] {
            let copy = CUDA_MEMCPY2D {
                srcXInBytes: 0,
                srcY: 0,
                srcMemoryType: CUmemorytype::CU_MEMORYTYPE_DEVICE,
                srcHost: ptr::null(),
                srcDevice: from,
                srcArray: ptr::null_mut(),
                srcPitch: pitch as usize,
                dstXInBytes: 0,
                dstY: 0,
                dstMemoryType: CUmemorytype::CU_MEMORYTYPE_HOST,
                dstHost: data[to..].as_mut_ptr().cast(),
                dstDevice: 0,
                dstArray: ptr::null_mut(),
                dstPitch: row,
                WidthInBytes: row,
                Height: rows,
            };
            check(cuda::cuMemcpy2D_v2(&copy), "copy to the host")?;
        }
        Ok::<(), DecodeError>(())
    })();
    check(cuvidUnmapVideoFrame64(state.decoder, device), "unmap")?;
    copied?;
    Ok(Picture {
        width: format.width,
        height: format.height,
        bit_depth: format.bit_depth,
        timestamp: info.timestamp,
        data,
    })
}
