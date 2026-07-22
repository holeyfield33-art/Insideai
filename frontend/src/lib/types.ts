/** Mirror of the backend websocket protocol (backend/app/streaming/protocol.py). */

export type TokenKind = "prompt" | "generated";

export interface TokenView {
  index: number;
  id: number;
  raw: string;
  text: string;
  kind: TokenKind;
}

export interface ModelInfo {
  model: string;
  family: string;
  chat_template: boolean;
  n_layer: number;
  n_head: number;
  n_embd: number;
  n_inner: number;
  n_positions: number;
  vocab_size: number;
  param_count: number;
  param_count_h: string;
  neuron_channels: number;
  neurons_per_channel: number;
  /** Pooled |W| block-means per layer — the model's real MLP wiring at
      diagram resolution (edges of the layered network view). */
  mlp_wiring: {
    in_nodes: number;
    hidden_nodes: number;
    layers: { w_in: number[][]; w_out: number[][] }[];
  };
  full_attention_seq_cap: number;
  max_prompt_tokens: number;
  max_new_tokens_cap: number;
  positional_kind: string;
  device: string;
  load_seconds: number;
}

export interface GenParams {
  max_new_tokens: number;
  temperature: number;
  top_k: number;
  top_p: number;
  repetition_penalty: number;
  seed: number | null;
}

export interface TopKEntry {
  id: number;
  token: string;
  raw: string;
  prob: number;
  logit: number;
}

export interface LogitsData {
  topk: TopKEntry[];
  entropy_bits: number;
  temperature_used: number;
  vocab_size: number;
}

export interface SampledToken {
  id: number;
  token: string;
  raw: string;
  prob: number;
  rank: number;
  strategy: string;
  is_eos: boolean;
}

/** One recorded run, as listed by the backend's GET /runs endpoint. */
export interface RunSummary {
  id: string;
  prompt: string;
  model: string | null;
  timestamp: string | null;
  steps: number;
}

export type ServerEvent =
  | { type: "server_mode"; live: boolean }
  | ({ type: "model_info" } & ModelInfo)
  | {
      type: "generation_start";
      prompt: string;
      prompt_tokens: number;
      truncated: boolean;
      template: boolean;
      speed: number;
      params: GenParams & { strategy: string };
    }
  | { type: "tokenize"; step: number; tokens: TokenView[]; seq_len: number }
  | {
      type: "embeddings";
      step: number;
      points: [number, number, number][];
      norms: number[];
      dim: number;
      method: string;
    }
  | {
      type: "positional";
      step: number;
      kind: string;
      dims_shown: number;
      norms: number[];
      sample: number[][];
    }
  | { type: "layer_start"; step: number; layer: number }
  | {
      /** Prefill: complete matrices for the whole prompt. */
      type: "attention";
      step: number;
      layer: number;
      mode: "full";
      /** Head-mean attention matrix, quantized 0..255 (divide by 255). */
      mean: number[][];
      /** Per-head attention row of the newest token, quantized 0..255. */
      last_row: number[][];
      head_entropy_bits: number[];
      /** Full per-head matrices while seq <= cap, else null. */
      heads: number[][][] | null;
      quantization: number;
    }
  | {
      /** Decode: the new token's attention row (client accumulates the
          causal matrix — earlier rows can't attend to later tokens). */
      type: "attention";
      step: number;
      layer: number;
      mode: "row";
      mean_row: number[];
      head_rows: number[][];
      head_entropy_bits: number[];
      quantization: number;
    }
  | {
      type: "ffn";
      step: number;
      layer: number;
      activations: number[];
      pooled_from: number;
      mean_abs: number;
      max: number;
      active_frac: number;
      /** Pre-MLP hidden state and MLP output, pooled to the diagram's
          input/output node counts. */
      input_pooled?: number[];
      output_pooled?: number[];
    }
  | {
      type: "layer_end";
      step: number;
      layer: number;
      hidden_norm: number;
      residual_delta: number;
    }
  | {
      /** unitarity-lab's real passive-mode telemetry: zeta_proxy is
          zeta_raw (one reading per generation step, shared across that
          step's layers), flagged comes from VAR's calibrated
          SpectralRuptureDetector on spectral_gap. */
      type: "anomaly";
      step: number;
      layer: number;
      zeta_proxy: number;
      flagged: boolean;
      source: string;
    }
  | ({ type: "logits"; step: number } & LogitsData)
  | ({ type: "sampled"; step: number } & SampledToken)
  | {
      type: "step_end";
      step: number;
      token: SampledToken;
      text: string;
      total_seq: number;
      compute_ms: number;
    }
  | {
      type: "generation_end";
      text: string;
      steps: number;
      reason: string;
      duration_s: number;
      tokens_per_s: number | null;
    }
  | { type: "error"; message: string }
  | { type: "pong"; t: number };

export type Stage =
  | "idle"
  | "tokenize"
  | "embeddings"
  | "positional"
  | "layers"
  | "logits"
  | "sampled"
  | "streaming"
  | "done"
  | "error";

export type Zone = "overview" | "embedding" | "attention" | "tower" | "neurons";
