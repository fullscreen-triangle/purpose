use candle_core::{Device, Tensor};

/// Builds a fixed-length (batch, seq_len) input/target pair from variable-
/// length token-id sequences: pads to the longest example in the batch
/// (right-padded with `pad_id`), targets are inputs shifted left by one
/// (next-token prediction), with the final position's target set to
/// `pad_id`. Shared by the from-scratch and pretrained training loops —
/// batching is architecture-agnostic, only the model differs.
pub fn build_batch(batch: &[&Vec<u32>], pad_id: u32, device: &Device) -> Option<(Tensor, Tensor)> {
    let max_len = batch.iter().map(|ids| ids.len()).max()?;
    if max_len < 2 {
        return None;
    }

    let mut input_flat = Vec::with_capacity(batch.len() * max_len);
    let mut target_flat = Vec::with_capacity(batch.len() * max_len);

    for ids in batch {
        let mut padded = (*ids).clone();
        padded.resize(max_len, pad_id);
        let mut targets: Vec<u32> = padded[1..].to_vec();
        targets.push(pad_id);

        input_flat.extend_from_slice(&padded);
        target_flat.extend_from_slice(&targets);
    }

    let input_ids = Tensor::from_vec(input_flat, (batch.len(), max_len), device).ok()?;
    let targets = Tensor::from_vec(target_flat, (batch.len(), max_len), device).ok()?;
    Some((input_ids, targets))
}
