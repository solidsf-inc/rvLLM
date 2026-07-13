//! Pinned device-to-host token transfer with guarded buffer reuse.

use rvllm_core::{Result, RvllmError, SampleCtx, SamplingError, TokenId};
use rvllm_mem::{Event, PinnedPool, Stream};

pub struct PinnedTokens<'s> {
    events: [Event<'s>; 2],
    pub(crate) pool: Option<PinnedPool<i32>>,
    stream: &'s Stream,
    in_flight: [bool; 2],
    poisoned: bool,
}

impl<'s> PinnedTokens<'s> {
    pub fn new(max_tokens: usize, stream: &'s Stream) -> Result<Self> {
        Ok(Self {
            events: [Event::new(stream)?, Event::new(stream)?],
            pool: Some(PinnedPool::new(max_tokens)?),
            stream,
            in_flight: [false; 2],
            poisoned: false,
        })
    }

    /// Queue a device-to-host copy and return a consume-once completion ticket.
    ///
    /// # Safety
    ///
    /// `src` must point to at least `num_tokens` readable device `i32` token
    /// IDs and remain valid until the returned ticket is waited or dropped.
    pub unsafe fn launch_dtoh(&mut self, src: u64, num_tokens: u32) -> Result<DtoHTicket<'_, 's>> {
        if self.poisoned {
            return Err(invalid("pinned transfer is poisoned after a failed fence"));
        }
        let count = num_tokens as usize;
        let buf_idx = self.pool_ref().write_idx();
        self.synchronize_buffer(buf_idx)?;
        if count > self.pool_mut().write_buf_mut().len() {
            return Err(invalid("num_tokens exceeds pinned-buffer capacity"));
        }
        self.in_flight[buf_idx] = true;
        let stream = self.stream.raw();
        // SAFETY: the returned ticket exclusively borrows `self`; every exit
        // path fences the stream or leaks the allocation before safe access or
        // destruction, and `src` is the caller's documented device input.
        if let Err(error) = unsafe {
            self.pool_mut()
                .write_buf_mut()
                .copy_from_device_async(src, count, stream)
        } {
            return self.recover_failed_queue(error);
        }
        if let Err(error) = self.events[buf_idx].record() {
            return self.recover_failed_queue(error);
        }
        self.pool_mut().flip();
        Ok(DtoHTicket {
            pool: self,
            buf_idx,
            num_tokens,
            completed: false,
        })
    }

    fn pool_ref(&self) -> &PinnedPool<i32> {
        self.pool.as_ref().expect("pinned pool exists until drop")
    }

    fn pool_mut(&mut self) -> &mut PinnedPool<i32> {
        self.pool.as_mut().expect("pinned pool exists until drop")
    }

    fn synchronize_buffer(&mut self, buf_idx: usize) -> Result<()> {
        if !self.in_flight[buf_idx] {
            return Ok(());
        }
        match self.events[buf_idx].synchronize() {
            Ok(()) => {
                self.in_flight[buf_idx] = false;
                Ok(())
            }
            Err(event_error) => match self.stream.fence() {
                Ok(()) => {
                    self.in_flight = [false; 2];
                    Err(event_error)
                }
                Err(fence_error) => {
                    self.poisoned = true;
                    Err(fence_error)
                }
            },
        }
    }

    fn recover_failed_queue<T>(&mut self, queue_error: RvllmError) -> Result<T> {
        match self.stream.fence() {
            Ok(()) => {
                self.in_flight = [false; 2];
                Err(queue_error)
            }
            Err(fence_error) => {
                self.poisoned = true;
                Err(fence_error)
            }
        }
    }
}

impl Drop for PinnedTokens<'_> {
    fn drop(&mut self) {
        if self.in_flight.iter().any(|in_flight| *in_flight) && self.stream.fence().is_err() {
            if let Some(pool) = self.pool.take() {
                core::mem::forget(pool);
            }
        }
    }
}

#[must_use = "DtoHTicket must be waited before reading copied tokens"]
pub struct DtoHTicket<'p, 's> {
    pool: &'p mut PinnedTokens<'s>,
    buf_idx: usize,
    num_tokens: u32,
    completed: bool,
}

impl<'p, 's> DtoHTicket<'p, 's> {
    pub fn wait(mut self) -> Result<Vec<TokenId>> {
        let result = self.pool.synchronize_buffer(self.buf_idx);
        self.completed = !self.pool.in_flight[self.buf_idx];
        result?;
        if self.pool.pool_ref().read_idx() != self.buf_idx {
            return Err(invalid("pinned-buffer state changed before completion"));
        }
        let count = self.num_tokens as usize;
        let raw = &self.pool.pool_ref().read_buf().as_slice()[..count];
        if raw.iter().any(|token| *token < 0) {
            return Err(invalid("device returned a negative token ID"));
        }
        Ok(raw.iter().map(|token| TokenId(*token as u32)).collect())
    }

    pub fn num_tokens(&self) -> u32 {
        self.num_tokens
    }
}

impl Drop for DtoHTicket<'_, '_> {
    fn drop(&mut self) {
        if !self.completed {
            let _ = self.pool.synchronize_buffer(self.buf_idx);
            self.completed = !self.pool.in_flight[self.buf_idx];
        }
    }
}

fn invalid(reason: &'static str) -> RvllmError {
    RvllmError::Sampling {
        err: SamplingError::InvalidParams {
            reason: reason.to_owned(),
        },
        ctx: SampleCtx {
            op: "token dtoh",
            stream: 0,
        },
    }
}

#[cfg(all(test, not(feature = "cuda")))]
mod tests {
    use super::*;

    #[test]
    fn empty_copy_completes() {
        let stream = Stream::host_stub();
        let mut pool = PinnedTokens::new(4, &stream).unwrap();
        let ticket = unsafe { pool.launch_dtoh(0, 0) }.unwrap();
        assert_eq!(ticket.num_tokens(), 0);
        assert!(ticket.wait().unwrap().is_empty());
    }

    #[test]
    fn host_build_does_not_report_a_device_copy() {
        let stream = Stream::host_stub();
        let mut pool = PinnedTokens::new(4, &stream).unwrap();
        assert!(unsafe { pool.launch_dtoh(1, 1) }.is_err());
    }

    #[test]
    fn capacity_is_checked_before_copy() {
        let stream = Stream::host_stub();
        let mut pool = PinnedTokens::new(1, &stream).unwrap();
        assert!(unsafe { pool.launch_dtoh(1, 2) }.is_err());
    }

    #[test]
    fn forgotten_ticket_cannot_make_a_buffer_reusable_while_in_flight() {
        let stream = Stream::host_stub();
        let mut pool = PinnedTokens::new(1, &stream).unwrap();
        core::mem::forget(unsafe { pool.launch_dtoh(0, 0) }.unwrap());
        unsafe { pool.launch_dtoh(0, 0) }.unwrap().wait().unwrap();
        unsafe { pool.launch_dtoh(0, 0) }.unwrap().wait().unwrap();
        assert_eq!(pool.in_flight, [false; 2]);
    }
}
