use anyhow::{anyhow, bail, Context, Result};
use clap::{Parser, ValueEnum};
use paramecia_engine::{
    Device, DeviceOffloadMode, Error, KvCacheQuantization, ModelEngineBuilder, TokenOutputStream,
};
use serde::Serialize;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tokenizers::Tokenizer;
use tokio::sync::mpsc;

const RUST_BOOK_CONTENT: &str = r#"Welcome to The Rust Programming Language, an introductory book about Rust. The Rust programming language helps you write faster, more reliable software. High-level ergonomics and low-level control are often at odds in programming language design; Rust challenges that conflict. Through balancing powerful technical capacity and a great developer experience, Rust gives you the option to control low-level details (such as memory usage) without all the hassle traditionally associated with such control.
Who Rust Is For

Rust is ideal for many people for a variety of reasons. Let's look at a few of the most important groups.
Teams of Developers

Rust is proving to be a productive tool for collaborating among large teams of developers with varying levels of systems programming knowledge. Low-level code is prone to various subtle bugs, which in most other languages can only be caught through extensive testing and careful code review by experienced developers. In Rust, the compiler plays a gatekeeper role by refusing to compile code with these elusive bugs, including concurrency bugs. By working alongside the compiler, the team can spend its time focusing on the program's logic rather than chasing down bugs.

Rust also brings contemporary developer tools to the systems programming world:

Cargo, the included dependency manager and build tool, makes adding, compiling, and managing dependencies painless and consistent across the Rust ecosystem.
The rustfmt formatting tool ensures a consistent coding style across developers.
The Rust Language Server powers integrated development environment (IDE) integration for code completion and inline error messages.

By using these and other tools in the Rust ecosystem, developers can be productive while writing systems-level code.
Students

Rust is for students and those who are interested in learning about systems concepts. Using Rust, many people have learned about topics like operating systems development. The community is very welcoming and happy to answer students' questions. Through efforts such as this book, the Rust teams want to make systems concepts more accessible to more people, especially those new to programming.
Companies

Hundreds of companies, large and small, use Rust in production for a variety of tasks, including command line tools, web services, DevOps tooling, embedded devices, audio and video analysis and transcoding, cryptocurrencies, bioinformatics, search engines, Internet of Things applications, machine learning, and even major parts of the Firefox web browser.
Open Source Developers

Rust is for people who want to build the Rust programming language, community, developer tools, and libraries. We'd love to have you contribute to the Rust language.
People Who Value Speed and Stability

Rust is for people who crave speed and stability in a language. By speed, we mean both how quickly Rust code can run and the speed at which Rust lets you write programs. The Rust compiler's checks ensure stability through feature additions and refactoring. This is in contrast to the brittle legacy code in languages without these checks, which developers are often afraid to modify. By striving for zero-cost abstractions-higher-level features that compile to lower-level code as fast as code written manually-Rust endeavors to make safe code be fast code as well.

The Rust language hopes to support many other users as well; those mentioned here are merely some of the biggest stakeholders. Overall, Rust's greatest ambition is to eliminate the trade-offs that programmers have accepted for decades by providing safety and productivity, speed and ergonomics. Give Rust a try, and see if its choices work for you.
Who This Book Is For

This book assumes that you've written code in another programming language, but it doesn't make any assumptions about which one. We've tried to make the material broadly accessible to those from a wide variety of programming backgrounds. We don't spend a lot of time talking about what programming is or how to think about it. If you're entirely new to programming, you would be better served by reading a book that specifically provides an introduction to programming.
How to Use This Book

In general, this book assumes that you're reading it in sequence from front to back. Later chapters build on concepts in earlier chapters, and earlier chapters might not delve into details on a particular topic but will revisit the topic in a later chapter.

You'll find two kinds of chapters in this book: concept chapters and project chapters. In concept chapters, you'll learn about an aspect of Rust. In project chapters, we'll build small programs together, applying what you've learned so far. Chapter 2, Chapter 12, and Chapter 21 are project chapters; the rest are concept chapters.

Chapter 1 explains how to install Rust, how to write a "Hello, world!" program, and how to use Cargo, Rust's package manager and build tool. Chapter 2 is a hands-on introduction to writing a program in Rust, having you build up a number-guessing game. Here, we cover concepts at a high level, and later chapters will provide additional detail. If you want to get your hands dirty right away, Chapter 2 is the place for that. If you're a particularly meticulous learner who prefers to learn every detail before moving on to the next, you might want to skip Chapter 2 and go straight to Chapter 3, which covers Rust features that are similar to those of other programming languages; then, you can return to Chapter 2 when you'd like to work on a project applying the details you've learned.

In Chapter 4, you'll learn about Rust's ownership system. Chapter 5 discusses structs and methods. Chapter 6 covers enums, match expressions, and the if let and let...else control flow constructs. You'll use structs and enums to make custom types.

In Chapter 7, you'll learn about Rust's module system and about privacy rules for organizing your code and its public application programming interface (API). Chapter 8 discusses some common collection data structures that the standard library provides: vectors, strings, and hash maps. Chapter 9 explores Rust's error-handling philosophy and techniques.

Chapter 10 digs into generics, traits, and lifetimes, which give you the power to define code that applies to multiple types. Chapter 11 is all about testing, which even with Rust's safety guarantees is necessary to ensure that your program's logic is correct. In Chapter 12, we'll build our own implementation of a subset of functionality from the grep command line tool that searches for text within files. For this, we'll use many of the concepts we discussed in the previous chapters.

Chapter 13 explores closures and iterators: features of Rust that come from functional programming languages. In Chapter 14, we'll examine Cargo in more depth and talk about best practices for sharing your libraries with others. Chapter 15 discusses smart pointers that the standard library provides and the traits that enable their functionality.

In Chapter 16, we'll walk through different models of concurrent programming and talk about how Rust helps you program in multiple threads fearlessly. In Chapter 17, we build on that by exploring Rust's async and await syntax, along with tasks, futures, and streams, and the lightweight concurrency model they enable.

Chapter 18 looks at how Rust idioms compare to object-oriented programming principles you might be familiar with. Chapter 19 is a reference on patterns and pattern matching, which are powerful ways of expressing ideas throughout Rust programs. Chapter 20 contains a smorgasbord of advanced topics of interest, including unsafe Rust, macros, and more about lifetimes, traits, types, functions, and closures.

In Chapter 21, we'll complete a project in which we'll implement a low-level multithreaded web server!"#;

#[cfg(feature = "qwen3next_80b_a3b")]
const DEFAULT_TOKENIZER_REPO: &str = "Qwen/Qwen3-Next-80B-A3B-Instruct";
#[cfg(not(feature = "qwen3next_80b_a3b"))]
const DEFAULT_TOKENIZER_REPO: &str = "Qwen/Qwen3.5-35B-A3B";

#[derive(Debug, Clone, Copy, ValueEnum)]
enum DeviceArg {
    Auto,
    Cpu,
    Cuda,
    Metal,
    Vulkan,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum OffloadArg {
    Auto,
    None,
    Experts,
    Down,
    Updown,
}

impl OffloadArg {
    fn to_mode(self) -> DeviceOffloadMode {
        match self {
            Self::Auto => DeviceOffloadMode::default(),
            Self::None => DeviceOffloadMode::FullGpu,
            Self::Experts => DeviceOffloadMode::ExpertsOnCpu,
            Self::Down => DeviceOffloadMode::DownProjectionsOnCpu,
            Self::Updown => DeviceOffloadMode::UpDownProjectionsOnCpu,
        }
    }
}

#[derive(Debug, Parser)]
#[command(name = "eval")]
#[command(
    about = "Measure prefill and generation speed on a ChatML prompt sourced from The Rust Book."
)]
struct Args {
    /// Path to local GGUF model weights.
    #[arg(long)]
    model_path: PathBuf,

    /// Path to local tokenizer.json file (overrides tokenizer_repo).
    #[arg(long)]
    tokenizer_path: Option<PathBuf>,

    /// HuggingFace tokenizer repo to download tokenizer.json from.
    #[arg(long, default_value = DEFAULT_TOKENIZER_REPO)]
    tokenizer_repo: String,

    /// Device selection for model execution.
    #[arg(long, value_enum, default_value_t = DeviceArg::Auto)]
    device: DeviceArg,

    /// Expert offload mode.
    #[arg(long, value_enum, default_value_t = OffloadArg::None)]
    offload: OffloadArg,

    /// KV-cache quantization mode (f16, bf16, q8_0, q4k, ...).
    #[arg(long, default_value = "q8_0")]
    kv_cache_quant: String,

    /// Number of Rust Book tokens to use in the ChatML system prompt.
    #[arg(long, default_value_t = 1000)]
    prefill_system_tokens: usize,

    /// Number of generated tokens to sample.
    #[arg(long, default_value_t = 100)]
    generate_tokens: usize,

    /// Temperature for token sampling.
    #[arg(long, default_value_t = 0.7)]
    temperature: f64,

    /// Top-p (nucleus) sampling threshold.
    #[arg(long)]
    top_p: Option<f64>,

    /// Top-k sampling limit.
    #[arg(long, default_value_t = 64)]
    top_k: usize,

    /// Number of tail samples for distribution reporting.
    #[arg(long, default_value_t = 16)]
    tail_samples: usize,

    /// RNG seed.
    #[arg(long, default_value_t = 299_792_458)]
    seed: u64,
}

#[derive(Debug, Serialize)]
struct EvalOutput {
    prefill_tokens_per_second: f64,
    generation_tokens_per_second: f64,
    generated_text: String,
}

#[tokio::main]
async fn main() {
    if let Err(err) = run(Args::parse()).await {
        eprintln!("eval failed: {err:#}");
        std::process::exit(1);
    }
}

async fn run(args: Args) -> Result<()> {
    if args.prefill_system_tokens == 0 {
        bail!("--prefill-system-tokens must be greater than 0");
    }
    if args.generate_tokens == 0 {
        bail!("--generate-tokens must be greater than 0");
    }

    let mut builder = ModelEngineBuilder::new(&args.model_path)
        .offload_mode(args.offload.to_mode())
        .kv_cache_quant(parse_kv_cache_quant(&args.kv_cache_quant)?)
        .temperature(args.temperature)
        .top_k(args.top_k)
        .tail_samples(args.tail_samples)
        .seed(args.seed);

    if let Some(top_p) = args.top_p {
        builder = builder.top_p(top_p);
    }
    if let Some(path) = args.tokenizer_path.as_ref() {
        builder = builder.tokenizer_path(path);
    } else {
        builder = builder.tokenizer_repo(&args.tokenizer_repo);
    }

    builder = match args.device {
        DeviceArg::Auto => builder,
        DeviceArg::Cpu => builder.cpu(true),
        DeviceArg::Cuda => builder
            .device(Device::cuda_if_available(0).context("failed to initialize CUDA device 0")?),
        DeviceArg::Metal => {
            builder.device(Device::new_metal(0).context("failed to initialize Metal device 0")?)
        }
        DeviceArg::Vulkan => {
            builder.device(Device::new_vulkan(0).context("failed to initialize Vulkan device 0")?)
        }
    };

    let engine = builder
        .build()
        .map_err(|e| anyhow!("failed to build model engine: {e}"))?;

    let tokenizer = engine.tokenizer().clone();
    let system_prompt_tokens = select_system_prompt_tokens(&tokenizer, args.prefill_system_tokens)?;
    let prompt_tokens = build_chatml_prompt_tokens(&tokenizer, &system_prompt_tokens)?;

    let prefill_start = Instant::now();
    let prefill_progress_rx = engine
        .fill_context_tokens(&prompt_tokens)
        .await
        .map_err(|e| anyhow!("prefill failed to start: {e}"))?;
    let prefilled_tokens = collect_prefill_progress(prefill_progress_rx).await?;
    let prefill_elapsed = prefill_start.elapsed();

    let mut token_stream = TokenOutputStream::new(tokenizer);
    let mut generated_text = String::new();
    let generation_start = Instant::now();
    let mut generated_tokens = 0usize;

    for _ in 0..args.generate_tokens {
        let predicted = engine
            .predict_token()
            .await
            .map_err(|e| anyhow!("generation failed: {e}"))?;
        engine
            .commit_token(predicted.token_id)
            .await
            .map_err(|e| anyhow!("failed to commit token {}: {e}", predicted.token_id))?;

        if let Some(piece) = token_stream.next_token(predicted.token_id)? {
            generated_text.push_str(&piece);
        }

        generated_tokens += 1;
    }

    if let Some(rest) = token_stream.decode_rest()? {
        generated_text.push_str(&rest);
    }

    let generation_elapsed = generation_start.elapsed();

    let output = EvalOutput {
        prefill_tokens_per_second: tokens_per_second(prefilled_tokens as usize, prefill_elapsed),
        generation_tokens_per_second: tokens_per_second(generated_tokens, generation_elapsed),
        generated_text,
    };

    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

fn select_system_prompt_tokens(
    tokenizer: &Tokenizer,
    system_token_count: usize,
) -> Result<Vec<u32>> {
    let encoding = tokenizer
        .encode(RUST_BOOK_CONTENT, false)
        .map_err(|e| anyhow!("failed to tokenize Rust Book text: {e}"))?;
    let base_ids = encoding.get_ids();
    if base_ids.is_empty() {
        bail!("embedded Rust Book content tokenized to zero tokens");
    }

    let mut out = Vec::with_capacity(system_token_count);
    while out.len() < system_token_count {
        let remain = system_token_count - out.len();
        if remain >= base_ids.len() {
            out.extend_from_slice(base_ids);
        } else {
            out.extend_from_slice(&base_ids[..remain]);
        }
    }
    Ok(out)
}

fn build_chatml_prompt_tokens(tokenizer: &Tokenizer, system_tokens: &[u32]) -> Result<Vec<u32>> {
    const SYSTEM_PREFIX: &str = "<|im_start|>system\n";
    const CHAT_SUFFIX: &str = "\n<|im_end|>\n<|im_start|>user\nWhat is the Rust programming language?\n<|im_end|>\n<|im_start|>assistant\n";

    let mut prompt_tokens = encode_piece(tokenizer, SYSTEM_PREFIX)?;
    prompt_tokens.extend_from_slice(system_tokens);
    prompt_tokens.extend_from_slice(&encode_piece(tokenizer, CHAT_SUFFIX)?);
    Ok(prompt_tokens)
}

fn encode_piece(tokenizer: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    tokenizer
        .encode(text, false)
        .map(|encoding| encoding.get_ids().to_vec())
        .map_err(|e| anyhow!("failed to tokenize prompt piece: {e}"))
}

async fn collect_prefill_progress(
    mut rx: mpsc::Receiver<std::result::Result<u32, Error>>,
) -> Result<u32> {
    let mut latest = 0_u32;
    while let Some(update) = rx.recv().await {
        latest = update.map_err(|e| anyhow!("prefill failed: {e}"))?;
    }
    Ok(latest)
}

fn parse_kv_cache_quant(s: &str) -> Result<KvCacheQuantization> {
    KvCacheQuantization::from_str(s).ok_or_else(|| anyhow!("invalid --kv-cache-quant value '{s}'"))
}

fn tokens_per_second(tokens: usize, elapsed: Duration) -> f64 {
    let secs = elapsed.as_secs_f64();
    if secs <= 0.0 {
        0.0
    } else {
        tokens as f64 / secs
    }
}
