//! Ctl subcommand — run a WASM controller component.

use anyhow::Result;

use crate::args::ControlArgs;

pub fn run(args: &ControlArgs) -> Result<()> {
    let checkpoint_dir = args
        .checkpoint_dir
        .clone()
        .unwrap_or_else(|| "/tmp/paramecia-checkpoints".to_string());

    paramecia_host::run_host_cli(paramecia_host::HostOptions {
        controller: args.controller.clone(),
        cpu: args.cpu,
        offload: args.offload.clone(),
        no_kv_quant: args.no_kv_quant,
        model_id: args.model_id.clone(),
        model_file: args.model_file.clone(),
        tokenizer_repo: args.tokenizer_repo.clone(),
        model_path: args.model_path.clone(),
        tokenizer_path: args.tokenizer_path.clone(),
        snapshot_dir: args.snapshot_dir.clone(),
        top_k: args.top_k,
        tail_samples: args.tail_samples,
        temperature: args.temperature,
        top_p: args.top_p,
        seed: args.seed,
        layer_split: args.layer_split.clone(),
        n_speculative_inference: args.n_speculative_inference,
        n_speculative_training: args.n_speculative_training,
        commit_mtp_override: args.commit_mtp_override,
        repeat_penalty: 1.0,
        presence_penalty: 0.0,
        penalty_last_n: 128,
        minibatch_size: args.minibatch_size,
        training_lr: args.training_lr,
        training_epsilon: args.training_epsilon,
        training_perturbation_mode: args.training_perturbation_mode.clone(),
        checkpoint_dir,
        n_grad_steps: args.n_grad_steps,
        optimize_tensors: args.optimize_tensors.clone(),
        clip_threshold: args.clip_threshold,
        lb_loss: args.lb_loss,
        z_loss: args.z_loss,
        interactive_endpoint: None,
        quiet: false,
    })
}
