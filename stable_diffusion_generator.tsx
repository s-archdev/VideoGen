import React, { useState, useEffect, useRef } from 'react';
import { Download, Settings, Wand2, Image, Loader2 } from 'lucide-react';

const StableDiffusionGenerator = () => {
  const [prompt, setPrompt] = useState('');
  const [negativePrompt, setNegativePrompt] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedImage, setGeneratedImage] = useState(null);
  const [settings, setSettings] = useState({
    steps: 20,
    cfgScale: 7.5,
    width: 512,
    height: 512,
    seed: -1
  });
  const [showSettings, setShowSettings] = useState(false);
  const canvasRef = useRef(null);

  // Simplified noise generation for demonstration
  const generateNoise = (width, height) => {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    const imageData = ctx.createImageData(width, height);
    
    for (let i = 0; i < imageData.data.length; i += 4) {
      const noise = Math.random() * 255;
      imageData.data[i] = noise;     // R
      imageData.data[i + 1] = noise; // G
      imageData.data[i + 2] = noise; // B
      imageData.data[i + 3] = 255;   // A
    }
    
    ctx.putImageData(imageData, 0, 0);
    return canvas.toDataURL();
  };

  // Simulate the diffusion process with animated noise reduction
  const simulateDiffusion = async (prompt, steps, width, height) => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    canvas.width = width;
    canvas.height = height;

    // Start with pure noise
    let imageData = ctx.createImageData(width, height);
    
    for (let step = 0; step < steps; step++) {
      // Simulate denoising process
      const progress = step / steps;
      const noiseLevel = 1 - progress;
      
      for (let i = 0; i < imageData.data.length; i += 4) {
        const x = (i / 4) % width;
        const y = Math.floor((i / 4) / width);
        
        // Create simple patterns based on prompt keywords
        let r = 128, g = 128, b = 128;
        
        if (prompt.toLowerCase().includes('sunset') || prompt.toLowerCase().includes('orange')) {
          r = 255 * (1 - noiseLevel) + Math.random() * 255 * noiseLevel;
          g = 165 * (1 - noiseLevel) + Math.random() * 255 * noiseLevel;
          b = 0 * (1 - noiseLevel) + Math.random() * 255 * noiseLevel;
        } else if (prompt.toLowerCase().includes('ocean') || prompt.toLowerCase().includes('blue')) {
          r = 30 * (1 - noiseLevel) + Math.random() * 255 * noiseLevel;
          g = 144 * (1 - noiseLevel) + Math.random() * 255 * noiseLevel;
          b = 255 * (1 - noiseLevel) + Math.random() * 255 * noiseLevel;
        } else if (prompt.toLowerCase().includes('forest') || prompt.toLowerCase().includes('green')) {
          r = 34 * (1 - noiseLevel) + Math.random() * 255 * noiseLevel;
          g = 139 * (1 - noiseLevel) + Math.random() * 255 * noiseLevel;
          b = 34 * (1 - noiseLevel) + Math.random() * 255 * noiseLevel;
        } else {
          // Create abstract patterns for other prompts
          const centerX = width / 2;
          const centerY = height / 2;
          const distance = Math.sqrt((x - centerX) ** 2 + (y - centerY) ** 2);
          const normalized = distance / Math.sqrt(centerX ** 2 + centerY ** 2);
          
          r = (Math.sin(normalized * Math.PI + progress * 2) * 127 + 128) * (1 - noiseLevel) + Math.random() * 255 * noiseLevel;
          g = (Math.cos(normalized * Math.PI + progress) * 127 + 128) * (1 - noiseLevel) + Math.random() * 255 * noiseLevel;
          b = (Math.sin(normalized * Math.PI * 2 - progress) * 127 + 128) * (1 - noiseLevel) + Math.random() * 255 * noiseLevel;
        }
        
        imageData.data[i] = Math.max(0, Math.min(255, r));
        imageData.data[i + 1] = Math.max(0, Math.min(255, g));
        imageData.data[i + 2] = Math.max(0, Math.min(255, b));
        imageData.data[i + 3] = 255;
      }
      
      ctx.putImageData(imageData, 0, 0);
      
      // Small delay to show the denoising process
      await new Promise(resolve => setTimeout(resolve, 50));
    }
    
    return canvas.toDataURL();
  };

  const generateImage = async () => {
    if (!prompt.trim()) return;
    
    setIsGenerating(true);
    setGeneratedImage(null);
    
    try {
      // Simulate the stable diffusion process
      const result = await simulateDiffusion(
        prompt, 
        settings.steps, 
        settings.width, 
        settings.height
      );
      
      setGeneratedImage(result);
    } catch (error) {
      console.error('Generation failed:', error);
    } finally {
      setIsGenerating(false);
    }
  };

  const downloadImage = () => {
    if (generatedImage) {
      const link = document.createElement('a');
      link.download = `stable_diffusion_${Date.now()}.png`;
      link.href = generatedImage;
      link.click();
    }
  };

  const randomizeSeed = () => {
    setSettings(prev => ({
      ...prev,
      seed: Math.floor(Math.random() * 1000000)
    }));
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900 p-4">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-white mb-2 flex items-center justify-center gap-3">
            <Wand2 className="text-purple-400" />
            Stable Diffusion Generator
          </h1>
          <p className="text-gray-300">Transform your imagination into stunning AI-generated art</p>
        </div>

        <div className="grid lg:grid-cols-2 gap-8">
          {/* Controls Panel */}
          <div className="space-y-6">
            <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-white/20">
              <h2 className="text-xl font-semibold text-white mb-4">Generation Controls</h2>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Prompt
                  </label>
                  <textarea
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    placeholder="A beautiful sunset over the ocean, digital art, highly detailed..."
                    className="w-full h-24 px-3 py-2 bg-white/10 border border-white/30 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 resize-none"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Negative Prompt
                  </label>
                  <textarea
                    value={negativePrompt}
                    onChange={(e) => setNegativePrompt(e.target.value)}
                    placeholder="blurry, low quality, distorted..."
                    className="w-full h-16 px-3 py-2 bg-white/10 border border-white/30 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 resize-none"
                  />
                </div>

                <div className="flex gap-3">
                  <button
                    onClick={generateImage}
                    disabled={isGenerating || !prompt.trim()}
                    className="flex-1 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 disabled:opacity-50 disabled:cursor-not-allowed text-white font-semibold py-3 px-6 rounded-lg transition-all duration-200 flex items-center justify-center gap-2"
                  >
                    {isGenerating ? (
                      <>
                        <Loader2 className="w-5 h-5 animate-spin" />
                        Generating...
                      </>
                    ) : (
                      <>
                        <Image className="w-5 h-5" />
                        Generate
                      </>
                    )}
                  </button>
                  
                  <button
                    onClick={() => setShowSettings(!showSettings)}
                    className="bg-white/10 hover:bg-white/20 text-white p-3 rounded-lg transition-colors duration-200"
                  >
                    <Settings className="w-5 h-5" />
                  </button>
                </div>
              </div>
            </div>

            {/* Settings Panel */}
            {showSettings && (
              <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-white/20">
                <h3 className="text-lg font-semibold text-white mb-4">Advanced Settings</h3>
                
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Steps: {settings.steps}
                    </label>
                    <input
                      type="range"
                      min="10"
                      max="50"
                      value={settings.steps}
                      onChange={(e) => setSettings(prev => ({...prev, steps: parseInt(e.target.value)}))}
                      className="w-full accent-purple-500"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      CFG Scale: {settings.cfgScale}
                    </label>
                    <input
                      type="range"
                      min="1"
                      max="20"
                      step="0.5"
                      value={settings.cfgScale}
                      onChange={(e) => setSettings(prev => ({...prev, cfgScale: parseFloat(e.target.value)}))}
                      className="w-full accent-purple-500"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">Width</label>
                    <select
                      value={settings.width}
                      onChange={(e) => setSettings(prev => ({...prev, width: parseInt(e.target.value)}))}
                      className="w-full px-3 py-2 bg-white/10 border border-white/30 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                    >
                      <option value={256}>256</option>
                      <option value={512}>512</option>
                      <option value={768}>768</option>
                      <option value={1024}>1024</option>
                    </select>
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">Height</label>
                    <select
                      value={settings.height}
                      onChange={(e) => setSettings(prev => ({...prev, height: parseInt(e.target.value)}))}
                      className="w-full px-3 py-2 bg-white/10 border border-white/30 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                    >
                      <option value={256}>256</option>
                      <option value={512}>512</option>
                      <option value={768}>768</option>
                      <option value={1024}>1024</option>
                    </select>
                  </div>
                </div>
                
                <div className="mt-4">
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Seed: {settings.seed === -1 ? 'Random' : settings.seed}
                  </label>
                  <div className="flex gap-2">
                    <input
                      type="number"
                      value={settings.seed === -1 ? '' : settings.seed}
                      onChange={(e) => setSettings(prev => ({...prev, seed: e.target.value ? parseInt(e.target.value) : -1}))}
                      placeholder="Random"
                      className="flex-1 px-3 py-2 bg-white/10 border border-white/30 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500"
                    />
                    <button
                      onClick={randomizeSeed}
                      className="bg-purple-600 hover:bg-purple-700 text-white px-4 py-2 rounded-lg transition-colors duration-200"
                    >
                      Random
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Output Panel */}
          <div className="space-y-6">
            <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-white/20">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-semibold text-white">Generated Image</h2>
                {generatedImage && (
                  <button
                    onClick={downloadImage}
                    className="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-lg transition-colors duration-200 flex items-center gap-2"
                  >
                    <Download className="w-4 h-4" />
                    Download
                  </button>
                )}
              </div>
              
              <div className="relative aspect-square bg-black/20 rounded-lg overflow-hidden">
                {isGenerating ? (
                  <div className="absolute inset-0 flex flex-col items-center justify-center">
                    <canvas
                      ref={canvasRef}
                      className="max-w-full max-h-full object-contain"
                    />
                    <div className="absolute bottom-4 left-4 right-4 bg-black/50 rounded-lg p-3">
                      <div className="flex items-center gap-2 text-white text-sm">
                        <Loader2 className="w-4 h-4 animate-spin" />
                        Denoising image... This is a simulation of the stable diffusion process
                      </div>
                    </div>
                  </div>
                ) : generatedImage ? (
                  <img
                    src={generatedImage}
                    alt="Generated artwork"
                    className="w-full h-full object-contain"
                  />
                ) : (
                  <div className="absolute inset-0 flex items-center justify-center text-gray-400">
                    <div className="text-center">
                      <Image className="w-16 h-16 mx-auto mb-4 opacity-50" />
                      <p>Your generated image will appear here</p>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Info Panel */}
            <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-white/20">
              <h3 className="text-lg font-semibold text-white mb-3">How It Works</h3>
              <div className="text-gray-300 text-sm space-y-2">
                <p>• <strong>Diffusion Process:</strong> Starts with random noise and gradually denoises it guided by your text prompt</p>
                <p>• <strong>CFG Scale:</strong> Controls how closely the model follows your prompt (higher = more faithful)</p>
                <p>• <strong>Steps:</strong> More steps = higher quality but slower generation</p>
                <p>• <strong>Seed:</strong> Controls randomness - same seed + prompt = same image</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default StableDiffusionGenerator;