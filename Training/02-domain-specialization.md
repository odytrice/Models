# Domain Specialization & F# Training Data

The distilled model targets full-stack web development with the following stack. Training data is biased heavily toward these domains, with F# receiving the most weight due to underrepresentation in base model training data.

## Training Data Composition (by domain and teacher)

| Domain | % of Data | Teacher | Rationale |
|--------|-----------|---------|-----------|
| **F# (core language)** | 15% | DeepSeek V3.2 | Largest corpus, best statically-typed language coverage |
| **F# (libraries: Giraffe, Akka.NET, FsToolkit, etc.)** | 10-15% | DeepSeek V3.2 | Same rationale; niche libraries need largest corpus |
| **Svelte 5 / SvelteKit 2** | 15-20% | Kimi K2.5 | Best frontend, best instruction following |
| **TypeScript** | 10% | Kimi K2.5 | Best SWE-bench Multilingual for JS/TS |
| **Docker / Kubernetes** | 8-10% | MiniMax M2.7 | Best Terminal-Bench 2 (57.0%), system-level understanding |
| **Agentic SWE tasks** | 15% | MiniMax M2.7 | Best SWE-Pro (56.22%), multi-step bug fixing |
| **.NET / ASP.NET Core** | 5% | DeepSeek V3.2 | Strong Java proxy scores for statically-typed .NET |
| **Long-context samples (128K-204K)** | 5-10% | Kimi K2.5 | Only teacher with 256K native context |
| **Cross-domain prompts** | 5% | Kimi K2.5 | Best instruction following (IFEval 94%) |
| **General coding/reasoning** | 5-10% | Mixed (all three) | Prevents catastrophic forgetting |

~20% of total F# training data is library-specific; ~80% covers general F# language patterns. The model needs strong core language fundamentals first -- idiomatic patterns, computation expressions, DU design -- before library-specific knowledge.

## Existing Datasets on HuggingFace (to supplement)

**Svelte:**
- `Dreamslol/svelte-5-sveltekit-2` -- Svelte 5 + SvelteKit 2 docs-based QA
- `relai-ai/svelte-standard` -- Svelte documentation QA
- `relai-ai/svelte-reasoning` -- Svelte reasoning dataset
- `oxide-lab/tauri2-svelte5-rust-instruct` -- Tauri + Svelte 5 instruction pairs

**F#, .NET, Docker, Kubernetes:** No existing instruction-tuning datasets found. All must be generated via teacher APIs.

---

## F# Training Data: Core Libraries

The model must be deeply trained on these F# libraries, extracted from production codebases (Orange, Bankr, Refirmity). All F# data generated via **DeepSeek V3.2**.

### F#-Specific Libraries

| Library | Purpose | Priority |
|---------|---------|----------|
| **FsToolkit.ErrorHandling** (+ TaskResult) | Railway-oriented programming, Result/Option computation expressions | Critical |
| **Akka.NET** (Akka, Akka.FSharp, Akka.Cluster, Akka.DependencyInjection, Akka.Bootstrap.Docker) | Actor model, clustering, concurrency | Critical |
| **Giraffe** (+ Giraffe.OpenApi, Giraffe.ViewEngine) | F# web framework on ASP.NET Core | Critical |
| **Thoth.Json.Net** (+ Thoth.Json.Giraffe) | JSON serialization for F# types | High |
| **FSharp.SystemTextJson** | System.Text.Json support for DUs, options | High |
| **FSharp.Control.AsyncSeq** | Async sequence processing | High |
| **FSharp.Control.Reactive** | Reactive extensions for F# | Medium |
| **FSharp.Data** | Type providers (JSON, CSV, HTML) | Medium |
| **linq2db** (+ linq2db.PostgreSQL) | LINQ-based data access from F# | High |
| **EntityFrameworkCore.FSharp** | EF Core extensions for F# | Medium |
| **FAKE** (Fake.Core.Target, Fake.IO) | F# build system | Medium |
| **Fantomas** | F# code formatting | Low |
| **FsUnit** / FsUnit.xUnit | F# test assertions | Medium |

### .NET / Infrastructure Libraries

| Library | Purpose | Priority |
|---------|---------|----------|
| **FluentMigrator** (+ Runner.Postgres) | Database schema migrations | High |
| **Npgsql** / Npgsql.EFCore.PostgreSQL | PostgreSQL data access | High |
| **Serilog** (+ AspNetCore, enrichers, sinks) | Structured logging | High |
| **Confluent.Kafka** | Event streaming / message bus | High |
| **Minio** | S3-compatible object storage | Medium |
| **Stripe.net** | Payment processing | Medium |
| **Ulid** | ULID identifier generation | Medium |
| **DocumentFormat.OpenXml** | DOCX document processing | Low |
| **PdfPig** | PDF text extraction | Low |
| **CsvHelper** | CSV processing | Low |

---

## F# Training Topics

**Core language (must cover deeply):**
- Discriminated unions, pattern matching, active patterns
- Computation expressions (async, task, result, custom CEs)
- Type providers, units of measure
- Records, anonymous records, value types
- Pipe operators, composition, currying/partial application
- Seq/List/Array module functions, collection pipelines

**F# for web (Giraffe + ASP.NET Core):**
- HttpHandler composition with `>=>` operator
- Route handlers, model binding, content negotiation
- Giraffe.ViewEngine for server-side HTML
- OpenAPI generation with Giraffe.OpenApi
- JWT authentication middleware
- Dependency injection with F# partial application

**F# + Akka.NET (actor model):**
- Actor creation with `spawn`, `actorOf`
- Message passing, `mailbox.Receive()`
- Akka.Cluster for distributed systems
- Akka.DependencyInjection with ASP.NET Core DI
- Supervision strategies, actor lifecycle
- Akka.Bootstrap.Docker for containerized deployments

**F# error handling (FsToolkit.ErrorHandling):**
- `result { }` and `taskResult { }` computation expressions
- Railway-oriented programming patterns
- `Result.map`, `Result.bind`, `Result.mapError`
- `Option` CE and `option { }` blocks
- Combining Result with async/task workflows
- Making illegal states unrepresentable with types

**F# data access:**
- linq2db queries from F#, type-safe LINQ
- EF Core with F# (EntityFrameworkCore.FSharp)
- FluentMigrator migrations in F#
- Npgsql direct ADO.NET access
- Thoth.Json.Net for JSON encode/decode

**F# domain modeling:**
- Domain-driven design with F# types
- Making illegal states unrepresentable
- Event sourcing patterns in F#

---

## Example Prompts

**Cross-domain prompts (high value -- generated by Kimi K2.5):**
- "Build a Giraffe API with FsToolkit.ErrorHandling railway-oriented validation, containerized with Docker"
- "Create Akka.NET actors for background job processing in an F# ASP.NET Core app deployed to K8s"
- "Model a domain with F# discriminated unions, expose via Giraffe, consume from SvelteKit TypeScript frontend"
- "Write FluentMigrator migrations for a PostgreSQL schema used by linq2db in F#"
- "Dockerize an F# Giraffe server with multi-stage build and deploy to Kubernetes with Helm"

**Agentic SWE prompts (generated by MiniMax M2.7):**
- "Here's a bug report against this F# Giraffe codebase -- diagnose and fix it"
- "Refactor this SvelteKit app to use server-side load functions"
- "This Dockerfile has a layer caching issue causing slow builds -- fix it"
- "Debug this Kubernetes deployment that's crash-looping due to misconfigured ConfigMap"
- "This Akka.NET actor cluster is losing messages under load -- analyze and fix the supervision strategy"
