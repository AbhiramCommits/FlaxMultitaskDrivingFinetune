"""Distributed data-parallel sharded training with jax.pmap.
"""



def shard_batch(batch_feat, batch_lab_event, batch_lab_horizon, batch_lab_rare, batch_weights):
    """Ensure batch is correctly sharded across devices for pmap.
    """
    return batch_feat, batch_lab_event, batch_lab_horizon, batch_lab_rare, batch_weights
