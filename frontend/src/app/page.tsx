'use client'

import { useState, useRef, useEffect } from 'react';
import { UploadCloud, ShieldAlert, ShieldCheck, X, Image as ImageIcon, Loader2, CheckCircle2, Shield, Activity, Cpu, ArrowRight, Zap, Target, LockKeyhole, Layers, Search, Server, Globe, Download, FileJson } from 'lucide-react';

type ModalData = {
  title: string;
  icon: React.ElementType;
  description: string;
  list?: string[];
  explanation?: string;
  extra?: React.ReactNode;
};

const Modal = ({ isOpen, onClose, data }: { isOpen: boolean, onClose: () => void, data: ModalData | null }) => {
  useEffect(() => {
    const handleEsc = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    if (isOpen) {
      document.body.style.overflow = 'hidden';
      window.addEventListener('keydown', handleEsc);
    } else {
      document.body.style.overflow = 'unset';
    }
    return () => {
      document.body.style.overflow = 'unset';
      window.removeEventListener('keydown', handleEsc);
    };
  }, [isOpen, onClose]);

  if (!isOpen || !data) return null;

  return (
    <div 
      className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-sm animate-in fade-in duration-200"
      onClick={onClose}
    >
      <div 
        className="w-full max-w-lg bg-slate-900 border border-slate-700/60 rounded-2xl shadow-2xl overflow-hidden flex flex-col relative animate-in zoom-in-95 duration-200 max-h-[90vh]"
        onClick={(e) => e.stopPropagation()}
      >
        <button 
          onClick={onClose}
          className="absolute top-4 right-4 p-2 text-slate-400 hover:text-white hover:bg-slate-800 rounded-full transition-colors focus:outline-none z-10"
        >
          <X className="w-5 h-5" />
        </button>
        
        <div className="p-6 md:p-8 overflow-y-auto">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 bg-blue-500/10 rounded-lg border border-blue-500/20 flex-shrink-0">
              <data.icon className="w-6 h-6 text-blue-400" />
            </div>
            <h3 className="text-xl font-bold text-white pr-6">{data.title}</h3>
          </div>
          
          <p className="text-slate-300 leading-relaxed mb-6">
            {data.description}
          </p>
          
          {data.list && (
            <ul className="space-y-3 mb-6">
              {data.list.map((item, i) => (
                <li key={i} className="flex items-start gap-3 text-sm text-slate-300">
                  <CheckCircle2 className="w-4 h-4 text-blue-500 mt-0.5 flex-shrink-0" />
                  <span>{item}</span>
                </li>
              ))}
            </ul>
          )}
          
          {data.explanation && (
            <p className="text-sm text-slate-400 mb-6 border-l-2 border-slate-700 pl-4 py-1">
              {data.explanation}
            </p>
          )}

          {data.extra}
        </div>
        
        <div className="px-6 py-4 bg-slate-950/50 border-t border-slate-800/60 flex justify-end flex-shrink-0">
          <button 
            onClick={onClose}
            className="px-6 py-2 bg-slate-800 hover:bg-slate-700 text-white text-sm font-semibold rounded-lg transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500/50"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<{prediction: string, confidence: number, device?: string} | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  
  const [activeModal, setActiveModal] = useState<ModalData | null>(null);
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (file) {
      const objectUrl = URL.createObjectURL(file);
      setPreview(objectUrl);
      return () => URL.revokeObjectURL(objectUrl);
    } else {
      setPreview(null);
    }
  }, [file]);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelection(e.dataTransfer.files[0]);
    }
  };

  const handleFileSelection = (selectedFile: File) => {
    setError(null);
    setResult(null);
    if (!['image/jpeg', 'image/png', 'image/webp', 'image/jpg'].includes(selectedFile.type)) {
      setError("Unsupported file format. Please upload JPG, PNG, or WEBP.");
      return;
    }
    setFile(selectedFile);
  };

  const clearSelection = () => {
    setFile(null);
    setResult(null);
    setError(null);
  };

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const res = await fetch(`${apiUrl}/api/predict`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        if (res.status === 422) {
          throw new Error("HTTP 422: Image rejected by Input Integrity & Anomaly Detection. The image failed the configured integrity/anomaly check.");
        }
        const errorData = await res.json().catch(() => null);
        throw new Error(errorData?.detail || 'Analysis failed. Make sure backend is running.');
      }

      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error occurred');
    } finally {
      setLoading(false);
    }
  };

  const scrollToAnalyzer = () => {
    document.getElementById('analyzer')?.scrollIntoView({ behavior: 'smooth' });
  };

  const downloadReport = () => {
    if (!result || !file) return;
    const reportData = {
      application: "DefendAI",
      result: result.prediction,
      confidence: result.confidence,
      model: "ResNet18",
      processor: result.device || "Unknown",
      inputIntegrity: "Verified",
      fileName: file.name,
      timestamp: new Date().toISOString()
    };
    const blob = new Blob([JSON.stringify(reportData, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = "defendai-analysis-report.json";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const downloadImage = () => {
    if (!file) return;
    const url = URL.createObjectURL(file);
    const a = document.createElement('a');
    a.href = url;
    a.download = file.name;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const openModal = (data: ModalData) => setActiveModal(data);
  const closeModal = () => setActiveModal(null);

  const securityCards = [
    {
      title: "AI Image Detection",
      icon: Search,
      desc: "ResNet18 deep-learning classification analyzes the uploaded image and returns a REAL/FAKE prediction with confidence.",
      modal: {
        title: "AI Image Detection",
        icon: Search,
        description: "DefendAI uses a ResNet18 deep-learning model to analyze uploaded images and classify them as REAL or FAKE.",
        list: [
          "Model: ResNet18",
          "Framework: PyTorch",
          "Input: 224 × 224 image",
          "Classes: FAKE / REAL",
          "Output: Prediction + confidence",
          "Hardware: CUDA when available"
        ],
        explanation: "Model performance is measured on a held-out evaluation dataset and may vary across unseen image sources and AI generators."
      }
    },
    {
      title: "Input Integrity & Anomaly Detection",
      icon: ShieldCheck,
      badge: "ACTIVE",
      desc: "DefendAI performs a lightweight integrity and anomaly screening step before deepfake inference. It rejects malformed, corrupted, unsupported, oversized, or abnormally low-variance inputs before they reach the detection model.",
      modal: {
        title: "Input Integrity & Anomaly Detection",
        icon: ShieldCheck,
        description: "DefendAI performs a lightweight integrity and anomaly screening step before deepfake inference. It rejects malformed, corrupted, unsupported, oversized, or abnormally low-variance inputs before they reach the detection model.",
        list: [
          "File format validation",
          "Maximum 10 MB upload validation",
          "Corrupted image detection",
          "Low-variance anomaly detection"
        ],
        explanation: "Current scope: basic input integrity and anomaly detection. Advanced adversarial defenses such as FGSM/PGD are not implemented in the current version.",
        extra: (
          <div className="flex flex-wrap gap-2 text-xs font-mono">
            <span className="px-2 py-1 bg-rose-500/10 text-rose-400 border border-rose-500/20 rounded">HTTP 400 — Invalid/corrupted image</span>
            <span className="px-2 py-1 bg-amber-500/10 text-amber-400 border border-amber-500/20 rounded">HTTP 413 — File too large</span>
            <span className="px-2 py-1 bg-orange-500/10 text-orange-400 border border-orange-500/20 rounded">HTTP 422 — Integrity/anomaly rejection</span>
          </div>
        )
      }
    },
    {
      title: "Confidence-Based Results",
      icon: Activity,
      desc: "Every prediction includes a confidence score and processing-device information to aid transparent human review.",
      modal: {
        title: "Confidence-Based Results",
        icon: Activity,
        description: "Each completed analysis returns the predicted class together with a confidence score and the processing device used for inference.",
        list: [
          "Prediction",
          "Confidence",
          "Processing device",
          "Integrity status"
        ]
      }
    }
  ];

  const workflowSteps = [
    { 
      num: "01", title: "Upload", desc: "User submits an image.",
      modal: {
        title: "Step 01: Upload", icon: UploadCloud,
        description: "User submits an image through the secure web interface.",
        list: ["Supports drag and drop", "Supports click to browse", "Accepts JPG, JPEG, PNG, WebP"]
      }
    },
    { 
      num: "02", title: "Validate", desc: "File type, size, and image integrity are checked.",
      modal: {
        title: "Step 02: Validate", icon: CheckCircle2,
        description: "The system validates the initial file properties.",
        list: ["File format verification", "Maximum 10 MB size limit", "Corrupted file detection"]
      }
    },
    { 
      num: "03", title: "Protect", desc: "Input Integrity & Anomaly Detection checks the image.",
      modal: {
        title: "Step 03: Protect", icon: Shield,
        description: "Input Integrity & Anomaly Detection runs mathematical checks on the image payload.",
        list: ["Low-variance anomaly detection", "Identifies zero-variance edge cases", "Rejects malformed tensors via HTTP 422"]
      }
    },
    { 
      num: "04", title: "Analyze", desc: "Image is preprocessed and passed to ResNet18.",
      modal: {
        title: "Step 04: Analyze", icon: Cpu,
        description: "The verified image is passed to the deep learning pipeline.",
        list: ["ImageNet normalization", "Resized to 224x224 pixels", "Forward pass through ResNet18 architecture"]
      }
    },
    { 
      num: "05", title: "Verify", desc: "DefendAI returns REAL/FAKE + confidence.",
      modal: {
        title: "Step 05: Verify", icon: Target,
        description: "DefendAI aggregates the model outputs and returns the result.",
        list: ["Returns REAL or FAKE prediction", "Includes softmax confidence score", "Reports processing hardware used (CUDA/CPU)"]
      }
    }
  ];

  const techCards = [
    {
      name: "PyTorch",
      modal: { title: "PyTorch", icon: Cpu, description: "Deep-learning framework used to run the ResNet18 model." }
    },
    {
      name: "ResNet18",
      modal: { title: "ResNet18", icon: Layers, description: "Convolutional neural network architecture used for REAL/FAKE image classification." }
    },
    {
      name: "FastAPI",
      modal: { title: "FastAPI", icon: Server, description: "Python backend framework providing the prediction API." }
    },
    {
      name: "Next.js",
      modal: { title: "Next.js", icon: Globe, description: "Frontend framework used to build the DefendAI web interface." }
    },
    {
      name: "CUDA",
      modal: { title: "CUDA", icon: Zap, description: "GPU acceleration used for model inference when a compatible NVIDIA GPU is available." }
    },
    {
      name: "OpenCV / NumPy",
      modal: { title: "OpenCV / NumPy", icon: Target, description: "Used for image processing, validation, and numerical operations." }
    }
  ];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 font-sans selection:bg-blue-500/30 overflow-x-hidden">
      <Modal isOpen={!!activeModal} onClose={closeModal} data={activeModal} />
      
      {/* Navigation */}
      <header className="sticky top-0 z-50 w-full border-b border-slate-800/60 bg-slate-950/80 backdrop-blur-xl">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2 text-xl font-bold text-white cursor-pointer">
            <Shield className="w-6 h-6 text-blue-500" />
            <span>Defend<span className="text-blue-500">AI</span></span>
          </div>
          <nav className="hidden md:flex items-center gap-8 text-sm font-medium text-slate-400">
            <a href="#analyzer" className="hover:text-white transition-colors">Detection</a>
            <a href="#security" className="hover:text-white transition-colors">Security</a>
            <a href="#how-it-works" className="hover:text-white transition-colors">How It Works</a>
            <a href="#technology" className="hover:text-white transition-colors">Technology</a>
          </nav>
          <div>
            <button onClick={scrollToAnalyzer} className="px-4 py-2 text-sm font-semibold bg-blue-600 hover:bg-blue-500 text-white rounded-lg transition-colors">
              Analyze Image
            </button>
          </div>
        </div>
      </header>

      <main className="flex-grow flex flex-col items-center justify-start relative w-full">
        
        {/* Dynamic Background */}
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-full max-w-7xl h-[600px] overflow-hidden -z-10 pointer-events-none opacity-40">
          <div className="absolute top-[-20%] left-[-10%] w-[50%] h-[80%] rounded-full bg-blue-600/20 blur-[120px]" />
          <div className="absolute top-[10%] right-[-10%] w-[50%] h-[80%] rounded-full bg-indigo-600/20 blur-[120px]" />
        </div>

        {/* Hero Section */}
        <section className="w-full max-w-5xl mx-auto px-6 pt-24 pb-16 lg:pt-32 lg:pb-24 flex flex-col items-center text-center">
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-blue-500/10 border border-blue-500/20 text-blue-400 text-sm font-medium mb-8">
            <Activity className="w-4 h-4" />
            Deepfake Detection & AI Integrity Verification
          </div>
          <h1 className="text-5xl md:text-7xl font-extrabold mb-8 text-white leading-[1.1]">
            Detect AI-Generated Images.<br />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-indigo-400">Before They Become a Threat.</span>
          </h1>
          <p className="text-lg md:text-xl text-slate-400 mb-10 max-w-3xl leading-relaxed">
            DefendAI analyzes uploaded images using deep-learning classification and input integrity checks to help identify synthetic or manipulated content.
          </p>
          <div className="flex flex-col sm:flex-row items-center gap-4">
            <button onClick={scrollToAnalyzer} className="w-full sm:w-auto px-8 py-4 bg-blue-600 hover:bg-blue-500 text-white font-semibold rounded-xl flex items-center justify-center gap-2 transition-all">
              Analyze an Image <ArrowRight className="w-5 h-5" />
            </button>
            <a href="#how-it-works" className="w-full sm:w-auto px-8 py-4 bg-slate-900 border border-slate-700 hover:bg-slate-800 text-white font-semibold rounded-xl flex items-center justify-center transition-all">
              Explore How It Works
            </a>
          </div>
        </section>

        {/* Analyzer Section */}
        <section id="analyzer" className="w-full max-w-4xl mx-auto px-6 py-20 scroll-mt-24 relative z-10">
          <div className="text-center mb-10">
            <h2 className="text-3xl font-bold text-white mb-3">Analyze an Image</h2>
            <p className="text-slate-400">Upload an image and let DefendAI evaluate its authenticity.</p>
          </div>

          <div className="backdrop-blur-xl bg-slate-900/60 p-8 rounded-3xl shadow-2xl border border-slate-800/80">
            
            {!file && (
              <div 
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
                className={`relative flex flex-col items-center justify-center w-full h-72 border-2 border-dashed rounded-2xl cursor-pointer transition-all duration-300 ease-out group
                  ${dragActive ? 'border-blue-500 bg-blue-500/10' : 'border-slate-700 hover:border-blue-500/50 hover:bg-slate-800/80'}
                `}
              >
                <input 
                  ref={fileInputRef}
                  type="file" 
                  className="hidden" 
                  accept=".jpg,.jpeg,.png,.webp"
                  onChange={(e) => {
                    if (e.target.files?.[0]) handleFileSelection(e.target.files[0]);
                  }}
                />
                <div className="flex flex-col items-center justify-center p-6 text-center space-y-4">
                  <div className={`p-5 rounded-full bg-slate-950/80 border border-slate-800 transition-transform duration-300 ${dragActive ? 'scale-110' : 'group-hover:scale-110'}`}>
                    <UploadCloud className="w-10 h-10 text-blue-400" />
                  </div>
                  <div>
                    <p className="text-xl font-semibold text-slate-200">
                      Drag & drop your image here
                    </p>
                    <p className="text-sm text-slate-500 mt-2">
                      or click to browse from your device
                    </p>
                  </div>
                  <div className="flex items-center gap-4 text-xs text-slate-500 mt-6 font-medium uppercase tracking-wider bg-slate-950/50 px-4 py-2 rounded-lg">
                    <span className="flex items-center gap-1"><ImageIcon className="w-4 h-4" /> JPG, PNG, WEBP</span>
                    <span className="w-1 h-1 rounded-full bg-slate-700" />
                    <span>Max 10 MB</span>
                  </div>
                </div>
              </div>
            )}

            {file && (
              <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
                <div className="relative rounded-2xl overflow-hidden mb-8 bg-slate-950/80 border border-slate-800 h-80 flex items-center justify-center">
                  {preview && (
                    <img src={preview} alt="Preview" className="max-h-full max-w-full object-contain" />
                  )}
                  <button 
                    onClick={clearSelection}
                    disabled={loading}
                    className="absolute top-4 right-4 p-2.5 bg-slate-900/90 hover:bg-rose-500 text-slate-300 hover:text-white rounded-full transition-all backdrop-blur-md z-10 disabled:opacity-50"
                    title="Remove image"
                  >
                    <X className="w-5 h-5" />
                  </button>
                </div>

                {error && (
                  <div className="mb-8 p-5 rounded-xl bg-rose-500/10 border border-rose-500/20 flex items-start gap-4 shadow-sm">
                    <ShieldAlert className="w-6 h-6 text-rose-500 flex-shrink-0 mt-0.5" />
                    <div className="flex flex-col">
                      <span className="text-rose-400 font-semibold mb-1">Analysis Error</span>
                      <p className="text-sm text-rose-300/90 leading-relaxed">{error}</p>
                    </div>
                  </div>
                )}

                {!result && (
                  <button 
                    onClick={handleUpload}
                    disabled={loading}
                    className="w-full relative group overflow-hidden rounded-xl bg-blue-600 hover:bg-blue-500 p-px font-semibold transition-all disabled:opacity-70 disabled:cursor-not-allowed"
                  >
                    <div className="relative flex items-center justify-center w-full h-full px-6 py-4 transition-all">
                      {loading ? (
                        <div className="flex flex-col items-center justify-center space-y-2">
                          <div className="flex items-center gap-2 text-sm text-blue-100 font-medium">
                            <CheckCircle2 className="w-4 h-4 text-emerald-400" /> <span>File validation</span>
                          </div>
                          <div className="flex items-center gap-2 text-sm text-blue-100 font-medium">
                            <CheckCircle2 className="w-4 h-4 text-emerald-400" /> <span>Input integrity check</span>
                          </div>
                          <div className="flex items-center gap-2 text-sm text-white font-semibold animate-pulse mt-1">
                            <Loader2 className="w-5 h-5 animate-spin" /> <span>Deepfake model inference</span>
                          </div>
                        </div>
                      ) : (
                        <span className="text-lg text-white flex items-center gap-2">
                          <ShieldCheck className="w-5 h-5" /> Verify Authenticity
                        </span>
                      )}
                    </div>
                  </button>
                )}

                {result && (
                  <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
                    <div className="p-5 rounded-xl bg-slate-950/50 border border-slate-800 flex flex-col sm:flex-row sm:items-center justify-around gap-4">
                      <div className="flex items-center gap-3 text-sm text-emerald-400 font-medium">
                        <CheckCircle2 className="w-5 h-5" /> <span>Input integrity verified</span>
                      </div>
                      <div className="hidden sm:block w-px h-8 bg-slate-800" />
                      <div className="flex items-center gap-3 text-sm text-emerald-400 font-medium">
                        <CheckCircle2 className="w-5 h-5" /> <span>Deepfake analysis completed</span>
                      </div>
                    </div>

                    <div className={`p-8 rounded-2xl border-2 relative overflow-hidden flex flex-col md:flex-row items-center gap-8 ${
                      result.prediction === 'REAL' 
                        ? 'bg-emerald-950/20 border-emerald-500/30' 
                        : 'bg-rose-950/20 border-rose-500/30'
                    }`}>
                      <div className="flex-shrink-0 flex items-center justify-center w-32 h-32 rounded-full bg-slate-950/50 border border-slate-800/50">
                        {result.prediction === 'REAL' ? (
                          <ShieldCheck className="w-16 h-16 text-emerald-500" />
                        ) : (
                          <ShieldAlert className="w-16 h-16 text-rose-500" />
                        )}
                      </div>
                      
                      <div className="flex-grow flex flex-col justify-center w-full text-center md:text-left">
                        <p className="text-xs uppercase tracking-widest font-bold text-slate-500 mb-2">Analysis Result</p>
                        <h2 className={`text-5xl font-black mb-4 ${
                          result.prediction === 'REAL' ? 'text-emerald-400' : 'text-rose-400'
                        }`}>
                          {result.prediction}
                        </h2>
                        
                        <div className="mb-4">
                          <div className="flex items-center justify-between text-sm mb-2 font-medium">
                            <span className="text-slate-400">Confidence Score</span>
                            <span className="text-white">{result.confidence}%</span>
                          </div>
                          <div className="w-full bg-slate-950 rounded-full h-2.5 overflow-hidden border border-slate-800">
                            <div 
                              className={`h-full rounded-full transition-all duration-1000 ease-out ${
                                result.prediction === 'REAL' ? 'bg-emerald-500' : 'bg-rose-500'
                              }`}
                              style={{ width: `${result.confidence}%` }}
                            />
                          </div>
                        </div>

                        <div className="flex flex-wrap gap-3 mt-4 justify-center md:justify-start">
                          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded bg-slate-900 border border-slate-800 text-xs font-mono text-slate-400">
                            <Target className="w-3.5 h-3.5" /> Engine: ResNet18
                          </span>
                          {result.device && (
                            <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded bg-slate-900 border border-slate-800 text-xs font-mono text-slate-400">
                              <Cpu className="w-3.5 h-3.5" /> Processor: {result.device.toUpperCase()}
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                    
                    <div className="flex flex-col gap-3">
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                        <button
                          onClick={downloadReport}
                          className="w-full py-3 px-4 flex items-center justify-center gap-2 text-sm font-semibold text-white bg-blue-600 hover:bg-blue-500 rounded-xl transition-colors shadow-sm"
                        >
                          <FileJson className="w-4 h-4" /> Download Analysis Report
                        </button>
                        <button
                          onClick={downloadImage}
                          className="w-full py-3 px-4 flex items-center justify-center gap-2 text-sm font-semibold text-slate-200 bg-slate-700 hover:bg-slate-600 rounded-xl transition-colors border border-slate-600 shadow-sm"
                        >
                          <Download className="w-4 h-4" /> Download Image
                        </button>
                      </div>
                      <button 
                        onClick={clearSelection}
                        className="w-full py-3 text-sm font-semibold text-slate-300 bg-slate-800 hover:bg-slate-700 rounded-xl transition-colors border border-slate-700 hover:border-slate-600 mt-2"
                      >
                        Analyze Another Image
                      </button>
                    </div>
                  </div>
                )}
              </div>
            )}
            
            {/* Inline Security Indicators for the upload box */}
            {!file && (
              <div className="mt-8 pt-6 border-t border-slate-800/60 flex flex-wrap justify-center gap-6">
                <div className="flex items-center gap-2 text-sm text-slate-400">
                  <CheckCircle2 className="w-4 h-4 text-blue-500" /> Format validation
                </div>
                <div className="flex items-center gap-2 text-sm text-slate-400">
                  <CheckCircle2 className="w-4 h-4 text-blue-500" /> Integrity verification
                </div>
                <div className="flex items-center gap-2 text-sm text-slate-400">
                  <CheckCircle2 className="w-4 h-4 text-blue-500" /> Deep-learning analysis
                </div>
              </div>
            )}
          </div>
        </section>

        {/* Security Section: Two Layers of Protection */}
        <section id="security" className="w-full bg-slate-900/40 py-24 border-y border-slate-800/40">
          <div className="max-w-7xl mx-auto px-6">
            <div className="text-center mb-16">
              <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">Two Layers of Protection</h2>
              <p className="text-slate-400 max-w-2xl mx-auto">DefendAI employs a dual-stage pipeline ensuring that inference is only executed on mathematically robust, verified data.</p>
            </div>
            <div className="grid md:grid-cols-3 gap-8">
              {securityCards.map((card, idx) => (
                <div 
                  key={idx} 
                  onClick={() => openModal(card.modal)}
                  className="bg-slate-950/50 p-8 rounded-2xl border border-slate-800/60 hover:border-blue-500/30 transition-all cursor-pointer hover:scale-[1.02] group relative overflow-hidden flex flex-col"
                >
                  {card.badge && (
                    <div className="absolute top-0 right-0 px-3 py-1 bg-blue-500/20 text-blue-400 text-xs font-bold rounded-bl-lg border-b border-l border-blue-500/30">
                      {card.badge}
                    </div>
                  )}
                  <div className="w-12 h-12 bg-blue-500/10 rounded-xl flex items-center justify-center mb-6 border border-blue-500/20">
                    <card.icon className="w-6 h-6 text-blue-400" />
                  </div>
                  <h3 className="text-xl font-bold text-white mb-3">{card.title}</h3>
                  <p className="text-slate-400 leading-relaxed text-sm flex-grow">{card.desc}</p>
                  <div className="mt-6 flex items-center text-sm font-semibold text-blue-400 group-hover:text-blue-300 transition-colors">
                    Learn more <ArrowRight className="w-4 h-4 ml-1 transition-transform group-hover:translate-x-1" />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* How It Works */}
        <section id="how-it-works" className="w-full max-w-7xl mx-auto px-6 py-24">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">How It Works</h2>
            <p className="text-slate-400 max-w-2xl mx-auto">A transparent, sequential pipeline designed for speed and reliability.</p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
            {workflowSteps.map((step, idx) => (
              <div 
                key={idx} 
                onClick={() => openModal(step.modal)}
                className="flex flex-col bg-slate-900/40 p-6 rounded-2xl border border-slate-800/60 relative cursor-pointer hover:bg-slate-800/50 hover:border-blue-500/30 transition-all group hover:-translate-y-1"
              >
                <span className="text-4xl font-black text-slate-800 group-hover:text-slate-700 transition-colors absolute top-4 right-4">{step.num}</span>
                <h4 className="text-lg font-bold text-white mb-2 mt-8">{step.title}</h4>
                <p className="text-sm text-slate-400 flex-grow">{step.desc}</p>
                <div className="mt-4 flex items-center text-xs font-semibold text-blue-400 opacity-0 group-hover:opacity-100 transition-opacity">
                  Learn more <ArrowRight className="w-3 h-3 ml-1" />
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Technology Section */}
        <section id="technology" className="w-full bg-slate-900/40 py-24 border-y border-slate-800/40">
          <div className="max-w-7xl mx-auto px-6 text-center">
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">Built on a Practical AI Pipeline</h2>
            <p className="text-slate-400 max-w-2xl mx-auto mb-12">Powered by industry-standard open-source frameworks and libraries.</p>
            <div className="flex flex-wrap justify-center gap-4 max-w-4xl mx-auto">
              {techCards.map((tech, idx) => (
                <div 
                  key={idx} 
                  onClick={() => openModal(tech.modal)}
                  className="group flex items-center gap-2 px-6 py-3 bg-slate-950/80 border border-slate-800 rounded-xl text-slate-300 font-medium font-mono text-sm shadow-sm hover:border-blue-500/50 hover:text-blue-400 transition-all cursor-pointer hover:bg-slate-900 hover:scale-105"
                >
                  {tech.name}
                  <span className="opacity-0 group-hover:opacity-100 transition-opacity">
                    <ArrowRight className="w-3 h-3" />
                  </span>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Trust / Security Section */}
        <section className="w-full bg-slate-950 py-24 border-b border-slate-800/60 relative overflow-hidden">
          {/* Subtle glow */}
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full max-w-4xl h-[300px] rounded-full bg-blue-900/10 blur-[100px] pointer-events-none" />
          
          <div className="max-w-7xl mx-auto px-6 relative z-10 text-center">
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-12">Designed for Digital Trust</h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
              {[
                { icon: Globe, label: "Social Media" },
                { icon: Layers, label: "News & Content Verification" },
                { icon: LockKeyhole, label: "Digital Identity Workflows" },
                { icon: ShieldCheck, label: "Image Authenticity Checks" }
              ].map((item, i) => (
                <div key={i} className="flex flex-col items-center justify-center p-6 bg-slate-900/60 rounded-2xl border border-slate-800">
                  <item.icon className="w-8 h-8 text-blue-500 mb-4" />
                  <span className="text-sm font-semibold text-slate-200">{item.label}</span>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Metrics Section */}
        <section className="w-full max-w-7xl mx-auto px-6 py-24">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">Project Performance</h2>
            <p className="text-slate-400 max-w-2xl mx-auto">Verified results from our robust formal evaluation phase.</p>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6 max-w-5xl mx-auto">
            {[
              { value: "88.10%", label: "Held-out Accuracy" },
              { value: "94.79%", label: "ROC-AUC" },
              { value: "90.19%", label: "Precision" },
              { value: "87.78%", label: "F1 Score" }
            ].map((metric, i) => (
              <div key={i} className="flex flex-col items-center justify-center p-8 bg-slate-900/40 rounded-2xl border border-slate-800/60">
                <span className="text-4xl md:text-5xl font-black text-white mb-2">{metric.value}</span>
                <span className="text-sm text-slate-400 uppercase tracking-widest font-semibold">{metric.label}</span>
              </div>
            ))}
          </div>
          <p className="text-center text-xs text-slate-500 mt-8">
            * Results measured on our held-out 2,000-image evaluation set.
          </p>
        </section>

        {/* Limitations */}
        <section className="w-full max-w-4xl mx-auto px-6 pb-24 text-center">
          <div className="p-6 rounded-2xl bg-slate-900/30 border border-slate-800/40 text-sm text-slate-500 leading-relaxed">
            <p className="mb-2"><strong className="text-slate-400">Important Limitations:</strong> Detection performance can vary across unseen image sources and AI generators.</p>
            <p>DefendAI&apos;s current integrity layer is a lightweight anomaly/integrity check rather than a sophisticated adversarial-defense system.</p>
          </div>
        </section>

      </main>

      {/* Footer */}
      <footer className="w-full border-t border-slate-800/60 bg-slate-950 pt-16 pb-8">
        <div className="max-w-7xl mx-auto px-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-12 mb-12">
            <div>
              <div className="flex items-center gap-2 text-xl font-bold text-white mb-4">
                <Shield className="w-6 h-6 text-blue-500" />
                <span>Defend<span className="text-blue-500">AI</span></span>
              </div>
              <p className="text-sm text-slate-500 max-w-xs">
                Deepfake Detection & AI Integrity Verification
              </p>
            </div>
            <div className="flex flex-col gap-3">
              <span className="text-white font-semibold mb-2">Navigation</span>
              <a href="#analyzer" className="text-sm text-slate-400 hover:text-white transition-colors w-fit">Detection</a>
              <a href="#security" className="text-sm text-slate-400 hover:text-white transition-colors w-fit">Security</a>
              <a href="#how-it-works" className="text-sm text-slate-400 hover:text-white transition-colors w-fit">How It Works</a>
              <a href="#technology" className="text-sm text-slate-400 hover:text-white transition-colors w-fit">Technology</a>
            </div>
            <div className="flex flex-col gap-3">
              <span className="text-white font-semibold mb-2">Notice</span>
              <p className="text-sm text-slate-500">
                Built as an engineering project for research and demonstration.
              </p>
            </div>
          </div>
          <div className="border-t border-slate-800/60 pt-8 flex flex-col md:flex-row items-center justify-between gap-4">
            <p className="text-xs text-slate-600">© 2026 DefendAI Project. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  );
}
