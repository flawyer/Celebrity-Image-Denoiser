"use client";
import { useState, useEffect, useMemo } from "react";
import {
  Image as ImageIcon,
  Sparkles,
  Upload,
  Download,
  Zap,
  BarChart3,
  Activity,
  Cpu,
  Workflow,
  Play,
  PlayCircle,
} from "lucide-react";

// ---- Types ----
type ModelType = "denoise" | "cgan" | "srgan" | "esrgan";

interface ModelResult {
  image: string | null;
  graph: string | null;
  processing: boolean;
  error: string | null;
}

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

export default function HomePage() {
  // Inputs
  const [originalImage, setOriginalImage] = useState<string | null>(null);
  const [originalFile, setOriginalFile] = useState<File | null>(null);

  const [condPreview, setCondPreview] = useState<string | null>(null);
  const [condFile, setCondFile] = useState<File | null>(null);

  // Per-model results
  const initialResult: ModelResult = useMemo(
    () => ({ image: null, graph: null, processing: false, error: null }),
    []
  );
  const [results, setResults] = useState<Record<ModelType, ModelResult>>({
    denoise: { ...initialResult },
    cgan: { ...initialResult },
    srgan: { ...initialResult },
    esrgan: { ...initialResult },
  });

  // UI state
  const [activeModel, setActiveModel] = useState<ModelType>("denoise");
  const [isClient, setIsClient] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [runningAll, setRunningAll] = useState(false);

  useEffect(() => setIsClient(true), []);

  // --- Helpers ---
  const buildQuery = (model: ModelType) => {
    const params = new URLSearchParams();
    params.set("model", model);
    if (model === "cgan") params.set("cgan_backend", "keras");
    return `${API_BASE}/enhance?${params.toString()}`;
  };

  const resetResults = (models?: ModelType[]) => {
    setResults((prev) => {
      const updated = { ...prev };
      (models ?? (Object.keys(prev) as ModelType[])).forEach((m) => {
        updated[m] = { image: null, graph: null, processing: false, error: null };
      });
      return updated;
    });
  };

  // --- Core: run one model ---
  const enhanceOne = async (model: ModelType) => {
    if (!originalFile) return;

    setResults((prev) => ({
      ...prev,
      [model]: { ...prev[model], processing: true, error: null, image: null, graph: null },
    }));

    try {
      const formData = new FormData();
      formData.append("file", originalFile);

      if (model === "cgan") {
        // Default label for demo and optional conditioning image
        formData.append("label", "5");
        if (condFile) formData.append("cond_file", condFile);
      }

      const response = await fetch(buildQuery(model), { method: "POST", body: formData });
      if (!response.ok) {
        const maybe = await response.json().catch(() => null);
        const msg = maybe?.detail || `Failed (${model}) HTTP ${response.status}`;
        throw new Error(msg);
      }

      const data = await response.json();
      // Back-end returns consistent keys across models
      const img = data.denoised_image_base64 || data.output_image_base64 || null;
      const graph = data.noise_graph_base64 || data.analysis_graph_base64 || null;

      setResults((prev) => ({
        ...prev,
        [model]: {
          image: img ? `data:image/png;base64,${img}` : null,
          graph: graph ? `data:image/png;base64,${graph}` : null,
          processing: false,
          error: null,
        },
      }));
    } catch (err: any) {
      setResults((prev) => ({
        ...prev,
        [model]: { ...prev[model], processing: false, error: err?.message || "Enhancement failed" },
      }));
    }
  };

  // --- Run all models in a pipeline (sequential to manage GPU/CPU) ---
  const allModels: ModelType[] = ["denoise", "cgan", "srgan", "esrgan"];

  const enhanceAll = async () => {
    if (!originalFile) return;
    setRunningAll(true);
    resetResults(allModels);
    for (const m of allModels) {
      // For cGAN without cond image, still run with label
      await enhanceOne(m);
    }
    setRunningAll(false);
  };

  // --- Upload handlers ---
  const handleImageUpload = (file: File) => {
    if (!file) return;
    setOriginalFile(file);
    setOriginalImage(URL.createObjectURL(file));
    // Note: No automatic API calls on upload. Use the Run buttons to start.
  };

  const onInputFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) handleImageUpload(f);
  };

  const onCondInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) {
      setCondFile(f);
      setCondPreview(URL.createObjectURL(f));
    }
  };

  const handleDrop = (event: React.DragEvent) => {
    event.preventDefault();
    setDragOver(false);
    const file = event.dataTransfer.files?.[0];
    if (file && file.type.startsWith("image/")) handleImageUpload(file);
  };
  const handleDragOver = (event: React.DragEvent) => {
    event.preventDefault();
    setDragOver(true);
  };
  const handleDragLeave = () => setDragOver(false);

  // --- Download helpers ---
  const downloadDataUrl = (dataUrl: string, filename: string) => {
    const link = document.createElement("a");
    link.href = dataUrl;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleDownloadImage = (model: ModelType) => {
    const url = results[model].image;
    if (url) downloadDataUrl(url, `${model}-output-${Date.now()}.png`);
  };

  const handleDownloadGraph = (model: ModelType) => {
    const url = results[model].graph;
    if (url) downloadDataUrl(url, `noise-analysis-${model}-${Date.now()}.png`);
  };

  // --- Clear cGAN cond when switching away ---
  useEffect(() => {
    if (activeModel !== "cgan") {
      setCondFile(null);
      setCondPreview(null);
    }
  }, [activeModel]);

  return (
    <main className="min-h-screen bg-black relative overflow-hidden">
      {/* Animated background */}
      <div className="absolute inset-0 bg-gradient-to-br from-purple-900/20 via-pink-900/10 to-cyan-900/20" />
      <div className="absolute inset-0">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl animate-pulse" />
        <div className="absolute bottom-1/4 right-1/4 w-80 h-80 bg-cyan-500/10 rounded-full blur-3xl animate-pulse delay-1000" />
        <div className="absolute top-1/2 left-1/2 w-64 h-64 bg-pink-500/10 rounded-full blur-3xl animate-pulse delay-500" />
      </div>

      {/* Floating particles */}
      {isClient && (
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          {[...Array(50)].map((_, i) => (
            <div
              key={i}
              className="absolute w-1 h-1 bg-white/20 rounded-full animate-float"
              style={{
                left: `${Math.random() * 100}%`,
                top: `${Math.random() * 100}%`,
                animationDelay: `${Math.random() * 5}s`,
                animationDuration: `${3 + Math.random() * 4}s`,
              }}
            />
          ))}
        </div>
      )}

      <div className="relative z-10 flex flex-col items-center justify-center min-h-screen px-4 py-10">
        {/* Header */}
        <div className="text-center mb-8 animate-fade-in">
          <h1 className="p-3 text-6xl md:text-7xl font-black mb-3 text-transparent bg-clip-text bg-gradient-to-r from-purple-400 via-pink-400 to-cyan-400 animate-gradient-x">
            AI Image Denoiser
          </h1>
          <div className="flex items-center justify-center gap-2 mb-4">
            <Zap className="w-6 h-6 text-yellow-400 animate-pulse" />
            <p className="text-lg text-gray-300 font-light">Run DenoiseGAN, cGAN, SRGAN, & ESRGAN with per-model analytics</p>
            <Zap className="w-6 h-6 text-yellow-400 animate-pulse" />
          </div>
          <div className="w-28 h-1 bg-gradient-to-r from-purple-400 to-cyan-400 mx-auto rounded-full" />
        </div>

        {/* Controls */}
        <div className="w-full max-w-7xl mb-8 grid gap-4 lg:grid-cols-2">
          <div className="bg-gray-900/70 border border-gray-700 rounded-xl p-4">
            <div className="flex items-center gap-2 mb-3 text-white font-semibold">
              <Workflow className="w-5 h-5 text-purple-400" />
              Pipeline
            </div>
            <div className="grid grid-cols-4 gap-2">
              {(["denoise", "cgan", "srgan", "esrgan"] as ModelType[]).map((m) => (
                <button
                  key={m}
                  onClick={() => setActiveModel(m)}
                  className={`px-3 py-2 rounded-lg border text-sm transition ${
                    activeModel === m
                      ? "border-purple-500 bg-purple-600/20 text-white"
                      : "border-gray-700 hover:border-purple-500 text-gray-300"
                  }`}
                  title={`Focus: ${m.toUpperCase()}`}
                >
                  {m.toUpperCase()}
                </button>
              ))}
            </div>
          </div>

          <div className="bg-gray-900/70 border border-gray-700 rounded-xl p-4">
            <div className="flex items-center gap-2 mb-3 text-white font-semibold">
              <Cpu className="w-5 h-5 text-emerald-400" />
              Run
            </div>
            <div className="flex flex-wrap gap-2">
              <button
                onClick={() => originalFile && enhanceOne(activeModel)}
                disabled={!originalFile || results[activeModel].processing || runningAll}
                className={`px-4 py-2 rounded-lg text-white transition flex items-center gap-2 ${
                  !originalFile || results[activeModel].processing || runningAll
                    ? "bg-gray-700 cursor-not-allowed"
                    : "bg-gradient-to-r from-purple-600 to-cyan-600 hover:from-purple-700 hover:to-cyan-700"
                }`}
                title={`Run ${activeModel.toUpperCase()} only`}
              >
                <Play className="w-4 h-4" />
                {results[activeModel].processing ? "Processing..." : `Run ${activeModel.toUpperCase()}`}
              </button>

              <button
                onClick={() => originalFile && enhanceAll()}
                disabled={!originalFile || runningAll}
                className={`px-4 py-2 rounded-lg text-white transition flex items-center gap-2 ${
                  !originalFile || runningAll
                    ? "bg-gray-700 cursor-not-allowed"
                    : "bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-700 hover:to-teal-700"
                }`}
                title="Run full pipeline (all models)"
              >
                <PlayCircle className="w-4 h-4" />
                {runningAll ? "Running Pipeline..." : "Run Full Pipeline"}
              </button>
            </div>
            {/* Status line */}
            <div className="mt-3 text-sm text-gray-300">
              {runningAll ? (
                <span className="text-amber-300">Pipeline executing sequentially…</span>
              ) : (
                <span className="text-gray-400">Select a model to re-run individually or run the full pipeline.</span>
              )}
            </div>
          </div>
        </div>

     

        {/* Upload Area */}
        <div
          className={`relative group transition-all duration-300 ${dragOver ? "scale-105" : "hover:scale-102"}`}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
        >
          <label
            htmlFor="fileInput"
            className={`cursor-pointer block transition-all duration-300 ${
              dragOver
                ? "bg-gradient-to-r from-purple-600/30 to-cyan-600/30 shadow-2xl shadow-purple-500/25"
                : "bg-gradient-to-r from-gray-800/50 to-gray-900/50 hover:from-purple-800/30 hover:to-cyan-800/30"
            } backdrop-blur-xl border border-gray-700/50 hover:border-purple-500/50 px-12 py-8 rounded-2xl shadow-xl hover:shadow-2xl hover:shadow-purple-500/20 flex flex-col items-center space-y-4`}
          >
            {/* Animated border */}
            <div className="absolute inset-0 rounded-2xl bg-gradient-to-r from-purple-500/20 via-pink-500/20 to-cyan-500/20 opacity-0 group-hover:opacity-100 transition-opacity duration-300 blur-sm" />

            <div className="relative z-10 flex flex-col items-center space-y-4">
              <div className="relative">
                <Upload className="w-12 h-12 text-purple-400 group-hover:text-purple-300 transition-colors duration-300" />
                <div className="absolute -inset-2 bg-purple-500/20 rounded-full blur-md opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
              </div>

              <div className="text-center">
                <span className="text-2xl font-bold text-white mb-2 block">Drop your image here</span>
                <span className="text-gray-400 text-lg">or click to Upload</span>
              </div>

              <div className="flex items-center space-x-2 text-sm text-gray-500">
                <span>•</span>
                <span>JPG, PNG, WebP</span>
                <span>•</span>
                <span>Max 50MB</span>
                <span>•</span>
              </div>
            </div>

            <input id="fileInput" type="file" accept="image/*" className="hidden" onChange={onInputFile} />
          </label>
        </div>

        {/* Results Section */}
        {originalImage && (
          <div className="mt-12 w-full max-w-7xl animate-slide-up">
            {/* Input preview */}
            <div className="grid grid-cols-1 gap-8 mb-10">
              <div className="group">
                <div className="relative">
                  <div className="absolute -inset-1 bg-gradient-to-r from-gray-600 to-gray-800 rounded-2xl blur-sm opacity-75" />
                  <div className="relative bg-gray-900 rounded-2xl p-6 border border-gray-700">
                    <div className="flex items-center justify-between mb-4">
                      <h3 className="text-2xl font-bold text-white flex items-center gap-2">
                        <ImageIcon className="w-6 h-6 text-gray-400" />
                        Original Image
                      </h3>
                      <span className="px-3 py-1 bg-gray-700 text-gray-300 rounded-full text-sm font-medium">Input</span>
                    </div>
                    <div className="relative overflow-hidden rounded-xl">
                      <img src={originalImage} alt="Original" className="w-full max-h-[500px] object-contain shadow-2xl transition-transform duration-300 group-hover:scale-105" />
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Per-model output grid */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 lg:gap-12">
              {(allModels as ModelType[]).map((m) => {
                const r = results[m];
                const busy = r.processing || (runningAll && !r.image && !r.error);
                return (
                  <div key={m} className="group">
                    <div className="relative">
                      <div className={`absolute -inset-1 rounded-2xl blur-sm opacity-75 ${
                        m === "denoise"
                          ? "bg-gradient-to-r from-slate-600 to-slate-800"
                          : m === "cgan"
                          ? "bg-gradient-to-r from-indigo-600 to-purple-700"
                          : m === "srgan"
                          ? "bg-gradient-to-r from-fuchsia-600 to-pink-600"
                          : "bg-gradient-to-r from-cyan-600 to-teal-600"
                      }`} />

                      <div className={`relative bg-gray-900 rounded-2xl p-6 border ${
                        m === "denoise"
                          ? "border-slate-400/40"
                          : m === "cgan"
                          ? "border-purple-400/40"
                          : m === "srgan"
                          ? "border-pink-400/40"
                          : "border-teal-400/40"
                      }`}>
                        <div className="flex items-center justify-between mb-4">
                          <h3 className="text-2xl font-bold text-white flex items-center gap-2">
                            <Sparkles className="w-6 h-6 text-purple-300" />
                            {m === "denoise" ? "Denoised" : m.toUpperCase()} Image
                          </h3>
                          <div className="flex gap-2">
                            <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                              busy
                                ? "bg-gray-700 text-gray-300"
                                : r.image
                                ? "bg-gradient-to-r from-purple-600 to-pink-600 text-white"
                                : r.error
                                ? "bg-red-700 text-white"
                                : "bg-gray-700 text-gray-300"
                            }`}>
                              {busy ? "Processing..." : r.error ? "Error" : r.image ? "Enhanced" : "Idle"}
                            </span>
                            {r.image && (
                              <button
                                onClick={() => handleDownloadImage(m)}
                                className="bg-gradient-to-r from-green-600 to-green-700 hover:from-green-700 hover:to-green-800 text-white px-3 py-1 rounded-full text-sm font-medium transition-all duration-300 hover:scale-105 flex items-center gap-1"
                                title={`Download ${m.toUpperCase()} image`}
                              >
                                <Download className="w-4 h-4" />
                                Download
                              </button>
                            )}
                          </div>
                        </div>

                        <div className="relative overflow-hidden rounded-xl min-h-[280px] flex items-center justify-center">
                          {busy ? (
                            <div className="w-full h-[320px] bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl flex flex-col items-center justify-center space-y-4">
                              <div className="relative">
                                <div className="w-16 h-16 border-4 border-purple-600/30 border-t-purple-400 rounded-full animate-spin" />
                                <div className="absolute top-2 left-2 w-12 h-12 border-4 border-pink-600/30 border-t-pink-400 rounded-full animate-spin animation-delay-150" />
                              </div>
                              <div className="text-center">
                                <p className="text-xl font-semibold text-white mb-2">Model running...</p>
                                <p className="text-gray-400">{m.toUpperCase()}</p>
                              </div>
                              <div className="flex space-x-1">
                                {[...Array(3)].map((_, i) => (
                                  <div key={i} className="w-2 h-2 bg-purple-400 rounded-full animate-pulse" style={{ animationDelay: `${i * 0.3}s` }} />
                                ))}
                              </div>
                            </div>
                          ) : r.image ? (
                            <img src={r.image} alt={`${m} output`} className="w-full max-h-[500px] object-contain shadow-2xl transition-transform duration-300 group-hover:scale-105" />
                          ) : r.error ? (
                            <div className="w-full h-[220px] bg-red-950/40 border border-red-700/40 rounded-xl flex items-center justify-center p-4">
                              <p className="text-red-300 text-sm text-center">{r.error}</p>
                            </div>
                          ) : (
                            <div className="w-full h-[220px] bg-gray-900/60 border border-gray-700/60 rounded-xl flex items-center justify-center">
                              <p className="text-gray-500">No output yet</p>
                            </div>
                          )}
                        </div>

                        {/* Graph + notes */}
                        {r.graph && !busy && (
                          <div className="mt-6">
                            <div className="relative group">
                              <div className="absolute -inset-1 bg-gradient-to-r from-orange-600 to-red-600 rounded-2xl blur-sm opacity-75" />
                              <div className="relative bg-gray-900 rounded-2xl p-4 border border-orange-500/50">
                                <div className="flex items-center justify-between mb-4">
                                  <div className="flex items-center gap-3">
                                    <div className="p-2 bg-gradient-to-r from-orange-600 to-red-600 rounded-lg">
                                      <BarChart3 className="w-6 h-6 text-white" />
                                    </div>
                                    <div>
                                      <h4 className="text-lg font-bold text-white">Noise Analysis — {m.toUpperCase()}</h4>
                                      <p className="text-gray-400 text-sm">Input vs Output difference insights</p>
                                    </div>
                                  </div>
                                  <button
                                    onClick={() => handleDownloadGraph(m)}
                                    className="bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white px-3 py-1 rounded-full text-sm font-medium transition-all duration-300 hover:scale-105 flex items-center gap-1"
                                    title={`Download ${m.toUpperCase()} analysis`}
                                  >
                                    <Download className="w-4 h-4" />
                                    Download
                                  </button>
                                </div>
                                <div className="relative overflow-hidden rounded-xl bg-white p-4">
                                  <img src={r.graph} alt={`${m} analysis`} className="w-full object-contain shadow-lg transition-transform duration-300 group-hover:scale-105" />
                                </div>
                                <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-3 text-sm">
                                  <div className="bg-gray-800/50 rounded-lg p-3 border border-gray-700">
                                    <h5 className="text-white font-semibold mb-1 flex items-center gap-2">
                                      <div className="w-3 h-3 bg-blue-500 rounded-full" /> Noise Map
                                    </h5>
                                    <p className="text-gray-400">Visualization of input − output differences.</p>
                                  </div>
                                  <div className="bg-gray-800/50 rounded-lg p-3 border border-gray-700">
                                    <h5 className="text-white font-semibold mb-1 flex items-center gap-2">
                                      <div className="w-3 h-3 bg-red-500 rounded-full" /> Error Map
                                    </h5>
                                    <p className="text-gray-400">Absolute difference highlighting stronger changes.</p>
                                  </div>
                                  <div className="bg-gray-800/50 rounded-lg p-3 border border-gray-700">
                                    <h5 className="text-white font-semibold mb-1 flex items-center gap-2">
                                      <div className="w-3 h-3 bg-green-500 rounded-full" /> Histogram
                                    </h5>
                                    <p className="text-gray-400">Distribution of pixel-wise differences.</p>
                                  </div>
                                </div>
                              </div>
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>

      <style jsx>{`
        @keyframes gradient-x {
          0%, 100% { background-size: 200% 200%; background-position: left center; }
          50% { background-size: 200% 200%; background-position: right center; }
        }
        @keyframes float {
          0%, 100% { transform: translateY(0px) rotate(0deg); }
          50% { transform: translateY(-10px) rotate(180deg); }
        }
        @keyframes fade-in {
          from { opacity: 0; transform: translateY(20px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes slide-up {
          from { opacity: 0; transform: translateY(40px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .animate-gradient-x { animation: gradient-x 3s ease infinite; }
        .animate-float { animation: float 3s ease-in-out infinite; }
        .animate-fade-in { animation: fade-in 0.6s ease-out; }
        .animate-slide-up { animation: slide-up 0.8s ease-out; }
        .hover\\:scale-102:hover { transform: scale(1.02); }
        .animation-delay-150 { animation-delay: 150ms; }
        .delay-300 { animation-delay: 300ms; }
      `}</style>
    </main>
  );
}
