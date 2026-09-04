'use client';

import {
  Download,
  FlaskConical,
  ImagePlus,
  RefreshCw,
  ScanLine,
  ShieldCheck,
} from 'lucide-react';
import { ChangeEvent, useCallback, useEffect, useRef, useState } from 'react';

type Analysis = {
  cells: number;
  coverage: number;
  averageArea: number;
  threshold: number;
};

declare global {
  interface Document {
    readonly modelContext?: {
      registerTool: (
        tool: {
          name: string;
          title: string;
          description: string;
          inputSchema: object;
          annotations: { readOnlyHint: boolean; untrustedContentHint: boolean };
          execute: (input: unknown) => Promise<object>;
        },
        options?: { signal?: AbortSignal },
      ) => void | Promise<void>;
    };
  }
}

const DEFAULT_IMAGE = '/sample-cell.png';
const MAX_FILE_SIZE = 15 * 1024 * 1024;

function otsuThreshold(values: Uint8Array): number {
  const histogram = new Uint32Array(256);
  for (const value of values) histogram[value] += 1;
  let totalIntensity = 0;
  for (let level = 0; level < 256; level += 1) {
    totalIntensity += level * histogram[level];
  }
  let backgroundWeight = 0;
  let backgroundIntensity = 0;
  let bestVariance = -1;
  let bestThreshold = 0;
  for (let level = 0; level < 256; level += 1) {
    backgroundWeight += histogram[level];
    if (backgroundWeight === 0) continue;
    const foregroundWeight = values.length - backgroundWeight;
    if (foregroundWeight === 0) break;
    backgroundIntensity += level * histogram[level];
    const backgroundMean = backgroundIntensity / backgroundWeight;
    const foregroundMean =
      (totalIntensity - backgroundIntensity) / foregroundWeight;
    const variance =
      backgroundWeight *
      foregroundWeight *
      (backgroundMean - foregroundMean) ** 2;
    if (variance > bestVariance) {
      bestVariance = variance;
      bestThreshold = level;
    }
  }
  return bestThreshold;
}

function buildLocalDeviation(
  gray: Uint8Array,
  width: number,
  height: number,
  radius: number,
): Uint8Array {
  const stride = width + 1;
  const integral = new Float64Array((width + 1) * (height + 1));
  for (let y = 0; y < height; y += 1) {
    let rowSum = 0;
    for (let x = 0; x < width; x += 1) {
      rowSum += gray[y * width + x];
      integral[(y + 1) * stride + x + 1] =
        integral[y * stride + x + 1] + rowSum;
    }
  }
  const deviation = new Uint8Array(gray.length);
  for (let y = 0; y < height; y += 1) {
    const top = Math.max(0, y - radius);
    const bottom = Math.min(height - 1, y + radius);
    for (let x = 0; x < width; x += 1) {
      const left = Math.max(0, x - radius);
      const right = Math.min(width - 1, x + radius);
      const sum =
        integral[(bottom + 1) * stride + right + 1] -
        integral[top * stride + right + 1] -
        integral[(bottom + 1) * stride + left] +
        integral[top * stride + left];
      const mean = sum / ((right - left + 1) * (bottom - top + 1));
      deviation[y * width + x] = Math.min(
        255,
        Math.round(Math.abs(gray[y * width + x] - mean)),
      );
    }
  }
  return deviation;
}

function keepComponents(
  rawMask: Uint8Array,
  width: number,
  height: number,
  minArea: number,
): { mask: Uint8Array; cells: number; totalArea: number } {
  const visited = new Uint8Array(rawMask.length);
  const cleanMask = new Uint8Array(rawMask.length);
  const queue = new Int32Array(rawMask.length);
  let cells = 0;
  let totalArea = 0;
  for (let start = 0; start < rawMask.length; start += 1) {
    if (!rawMask[start] || visited[start]) continue;
    let head = 0;
    let tail = 1;
    queue[0] = start;
    visited[start] = 1;
    while (head < tail) {
      const position = queue[head++];
      const x = position % width;
      const y = Math.floor(position / width);
      const neighbors = [
        x > 0 ? position - 1 : -1,
        x < width - 1 ? position + 1 : -1,
        y > 0 ? position - width : -1,
        y < height - 1 ? position + width : -1,
      ];
      for (const neighbor of neighbors) {
        if (neighbor >= 0 && rawMask[neighbor] && !visited[neighbor]) {
          visited[neighbor] = 1;
          queue[tail++] = neighbor;
        }
      }
    }
    if (tail >= minArea) {
      cells += 1;
      totalArea += tail;
      for (let index = 0; index < tail; index += 1) cleanMask[queue[index]] = 1;
    }
  }
  return { mask: cleanMask, cells, totalArea };
}

export default function Home() {
  const imageRef = useRef<HTMLImageElement>(null);
  const sourceCanvasRef = useRef<HTMLCanvasElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const objectUrlRef = useRef<string | null>(null);
  const [imageSrc, setImageSrc] = useState(DEFAULT_IMAGE);
  const [fileName, setFileName] = useState('LIVECell A172 example');
  const [radius, setRadius] = useState(9);
  const [sensitivity, setSensitivity] = useState(1);
  const [minArea, setMinArea] = useState(24);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [status, setStatus] = useState('Loading example…');
  const [error, setError] = useState('');

  const processImage = useCallback(() => {
    const image = imageRef.current;
    const sourceCanvas = sourceCanvasRef.current;
    const overlayCanvas = overlayCanvasRef.current;
    if (!image || !sourceCanvas || !overlayCanvas || !image.naturalWidth)
      return;
    setStatus('Analyzing image…');
    setError('');
    requestAnimationFrame(() => {
      const scale = Math.min(
        1,
        900 / Math.max(image.naturalWidth, image.naturalHeight),
      );
      const width = Math.max(1, Math.round(image.naturalWidth * scale));
      const height = Math.max(1, Math.round(image.naturalHeight * scale));
      sourceCanvas.width = width;
      sourceCanvas.height = height;
      overlayCanvas.width = width;
      overlayCanvas.height = height;
      const sourceContext = sourceCanvas.getContext('2d', {
        willReadFrequently: true,
      });
      const overlayContext = overlayCanvas.getContext('2d');
      if (!sourceContext || !overlayContext) {
        setError('This browser could not start the image analyzer.');
        setStatus('');
        return;
      }
      sourceContext.drawImage(image, 0, 0, width, height);
      const pixels = sourceContext.getImageData(0, 0, width, height);
      const gray = new Uint8Array(width * height);
      for (let index = 0; index < gray.length; index += 1) {
        const offset = index * 4;
        gray[index] = Math.round(
          pixels.data[offset] * 0.299 +
            pixels.data[offset + 1] * 0.587 +
            pixels.data[offset + 2] * 0.114,
        );
      }
      const deviation = buildLocalDeviation(gray, width, height, radius);
      const threshold = Math.max(
        1,
        Math.round(otsuThreshold(deviation) * sensitivity),
      );
      const rawMask = new Uint8Array(deviation.length);
      for (let index = 0; index < deviation.length; index += 1) {
        rawMask[index] = deviation[index] > threshold ? 1 : 0;
      }
      const result = keepComponents(rawMask, width, height, minArea);
      const overlay = new ImageData(
        new Uint8ClampedArray(pixels.data),
        width,
        height,
      );
      for (let index = 0; index < result.mask.length; index += 1) {
        if (!result.mask[index]) continue;
        const offset = index * 4;
        overlay.data[offset] = Math.round(overlay.data[offset] * 0.38 + 15);
        overlay.data[offset + 1] = Math.round(
          overlay.data[offset + 1] * 0.38 + 158,
        );
        overlay.data[offset + 2] = Math.round(
          overlay.data[offset + 2] * 0.38 + 170,
        );
      }
      overlayContext.putImageData(overlay, 0, 0);
      setAnalysis({
        cells: result.cells,
        coverage: (result.totalArea / result.mask.length) * 100,
        averageArea: result.cells ? result.totalArea / result.cells : 0,
        threshold,
      });
      setStatus('Analysis complete');
    });
  }, [minArea, radius, sensitivity]);

  useEffect(() => {
    const timer = window.setTimeout(processImage, 100);
    return () => window.clearTimeout(timer);
  }, [processImage]);

  useEffect(
    () => () => {
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    },
    [],
  );

  useEffect(() => {
    const context = document.modelContext;
    if (!context?.registerTool) return;
    const lifecycle = new AbortController();
    const registration = context.registerTool(
      {
        name: 'configure_cell_analysis',
        title: 'Configure cell analysis',
        description:
          'Adjust the visible TriboVision analysis settings for the current image.',
        inputSchema: {
          type: 'object',
          properties: {
            radius: { type: 'integer', minimum: 3, maximum: 20 },
            sensitivity: { type: 'number', minimum: 0.55, maximum: 1.65 },
            minimumRegionArea: { type: 'integer', minimum: 4, maximum: 120 },
          },
          additionalProperties: false,
        },
        annotations: { readOnlyHint: false, untrustedContentHint: false },
        async execute(input) {
          if (!input || typeof input !== 'object' || Array.isArray(input)) {
            throw new Error('Analysis settings must be an object.');
          }
          const values = input as Record<string, unknown>;
          const nextRadius = values.radius ?? radius;
          const nextSensitivity = values.sensitivity ?? sensitivity;
          const nextMinArea = values.minimumRegionArea ?? minArea;
          if (
            !Number.isInteger(nextRadius) ||
            Number(nextRadius) < 3 ||
            Number(nextRadius) > 20
          ) {
            throw new Error('radius must be an integer from 3 to 20.');
          }
          if (
            typeof nextSensitivity !== 'number' ||
            nextSensitivity < 0.55 ||
            nextSensitivity > 1.65
          ) {
            throw new Error('sensitivity must be a number from 0.55 to 1.65.');
          }
          if (
            !Number.isInteger(nextMinArea) ||
            Number(nextMinArea) < 4 ||
            Number(nextMinArea) > 120
          ) {
            throw new Error(
              'minimumRegionArea must be an integer from 4 to 120.',
            );
          }
          setRadius(Number(nextRadius));
          setSensitivity(nextSensitivity);
          setMinArea(Number(nextMinArea));
          await new Promise<void>((resolve) => {
            requestAnimationFrame(() => requestAnimationFrame(() => resolve()));
          });
          return {
            radius: Number(nextRadius),
            sensitivity: nextSensitivity,
            minimumRegionArea: Number(nextMinArea),
          };
        },
      },
      { signal: lifecycle.signal },
    );
    void Promise.resolve(registration).catch(() => undefined);
    return () => lifecycle.abort();
  }, [minArea, radius, sensitivity]);

  function chooseImage(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    if (!['image/png', 'image/jpeg', 'image/webp'].includes(file.type)) {
      setError('Please choose a PNG, JPEG, or WebP image.');
      return;
    }
    if (file.size > MAX_FILE_SIZE) {
      setError('Please choose an image smaller than 15 MB.');
      return;
    }
    if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    objectUrlRef.current = URL.createObjectURL(file);
    setFileName(file.name);
    setImageSrc(objectUrlRef.current);
    setAnalysis(null);
    setStatus('Loading image…');
    setError('');
  }

  function resetExample() {
    if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    objectUrlRef.current = null;
    setImageSrc(DEFAULT_IMAGE);
    setFileName('LIVECell A172 example');
    setRadius(9);
    setSensitivity(1);
    setMinArea(24);
    setAnalysis(null);
  }

  function downloadOverlay() {
    const canvas = overlayCanvasRef.current;
    if (!canvas) return;
    const link = document.createElement('a');
    link.download = `${fileName.replace(/\.[^.]+$/, '')}-tribovision-overlay.png`;
    link.href = canvas.toDataURL('image/png');
    link.click();
  }

  return (
    <main className="min-h-screen bg-[var(--background)] text-[var(--foreground)]">
      <header className="border-b border-white/10 px-5 py-4 sm:px-8">
        <div className="mx-auto flex max-w-[1500px] items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="grid h-9 w-9 place-items-center rounded-full border border-cyan-300/40 bg-cyan-300/10 text-cyan-200">
              <ScanLine size={19} aria-hidden="true" />
            </span>
            <div>
              <p className="font-mono text-base font-semibold tracking-[0.12em] text-white">
                TRIBOVISION
              </p>
              <p className="text-xs text-slate-400">Cell morphology explorer</p>
            </div>
          </div>
          <span className="rounded-full border border-amber-300/30 bg-amber-300/10 px-3 py-1 text-xs font-medium text-amber-200">
            Research prototype
          </span>
        </div>
      </header>

      <section className="mx-auto grid max-w-[1500px] gap-5 px-5 py-6 sm:px-8 lg:grid-cols-[290px_minmax(0,1fr)]">
        <aside className="lab-panel h-fit p-5 lg:sticky lg:top-6">
          <p className="eyebrow">Input</p>
          <h1 className="mt-1 text-2xl font-semibold tracking-tight text-white">
            Examine a cell image
          </h1>
          <p className="mt-2 text-sm leading-6 text-slate-400">
            Choose a microscope image. Processing stays in your browser.
          </p>

          <label className="upload-zone group mt-5">
            <ImagePlus className="text-cyan-200" size={23} aria-hidden="true" />
            <span className="font-medium text-slate-100">Upload an image</span>
            <span className="text-xs text-slate-500">
              PNG, JPEG or WebP · 15 MB max
            </span>
            <input
              className="sr-only"
              type="file"
              accept="image/png,image/jpeg,image/webp"
              onChange={chooseImage}
            />
          </label>
          <button
            className="secondary-button mt-3 w-full"
            onClick={resetExample}
          >
            <RefreshCw size={15} aria-hidden="true" /> Restore LIVECell example
          </button>

          <div className="mt-7 space-y-6 border-t border-white/10 pt-6">
            <Control
              label="Local contrast radius"
              output={`${radius}px`}
              min={3}
              max={20}
              value={radius}
              onChange={setRadius}
            />
            <Control
              label="Detection threshold"
              output={`${sensitivity.toFixed(2)}×`}
              min={0.55}
              max={1.65}
              step={0.05}
              value={sensitivity}
              onChange={setSensitivity}
            />
            <Control
              label="Minimum region size"
              output={`${minArea}px`}
              min={4}
              max={120}
              step={4}
              value={minArea}
              onChange={setMinArea}
            />
          </div>

          <div className="mt-7 flex gap-2 rounded-xl border border-cyan-300/15 bg-cyan-300/[0.04] p-3 text-xs leading-5 text-slate-400">
            <ShieldCheck className="mt-0.5 shrink-0 text-cyan-200" size={17} />
            Your image is analyzed locally and is not stored or sent anywhere.
          </div>
        </aside>

        <div className="min-w-0 space-y-5">
          <section className="lab-panel overflow-hidden">
            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-white/10 px-5 py-4">
              <div className="min-w-0">
                <p className="eyebrow">Live analysis</p>
                <p className="mt-1 truncate text-sm font-medium text-slate-200">
                  {fileName}
                </p>
              </div>
              <div className="flex items-center gap-3">
                <span
                  className="flex items-center gap-2 text-xs text-slate-400"
                  aria-live="polite"
                >
                  <span className="h-2 w-2 rounded-full bg-cyan-300 shadow-[0_0_12px_#67e8f9]" />{' '}
                  {status}
                </span>
                <button
                  className="primary-button"
                  onClick={downloadOverlay}
                  disabled={!analysis}
                >
                  <Download size={15} aria-hidden="true" /> Download overlay
                </button>
              </div>
            </div>
            {error ? (
              <p
                role="alert"
                className="border-b border-red-300/20 bg-red-300/10 px-5 py-3 text-sm text-red-200"
              >
                {error}
              </p>
            ) : null}
            {/* oxlint-disable-next-line next/no-img-element -- hidden decoder for user-selected object URLs */}
            <img
              ref={imageRef}
              src={imageSrc}
              alt=""
              className="hidden"
              onLoad={processImage}
              onError={() => {
                setError('The selected image could not be read.');
                setStatus('');
              }}
            />
            <div className="grid gap-px bg-white/10 xl:grid-cols-2">
              <Viewer
                number="01"
                label="Original image"
                canvasRef={sourceCanvasRef}
                ariaLabel="Original cell microscope image"
              />
              <Viewer
                number="02"
                label="Detected regions"
                canvasRef={overlayCanvasRef}
                ariaLabel="Cell image with cyan detection overlay"
              />
            </div>
          </section>

          <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <Metric
              label="Detected regions"
              value={analysis ? analysis.cells.toLocaleString() : '—'}
              unit="regions"
            />
            <Metric
              label="Image coverage"
              value={analysis ? analysis.coverage.toFixed(1) : '—'}
              unit="percent"
            />
            <Metric
              label="Mean region area"
              value={analysis ? analysis.averageArea.toFixed(0) : '—'}
              unit="pixels"
            />
            <Metric
              label="Contrast threshold"
              value={analysis ? analysis.threshold.toString() : '—'}
              unit="intensity"
            />
          </section>

          <section className="lab-panel grid gap-5 p-5 md:grid-cols-[auto_1fr]">
            <div className="grid h-11 w-11 place-items-center rounded-xl border border-violet-300/20 bg-violet-300/10 text-violet-200">
              <FlaskConical size={21} aria-hidden="true" />
            </div>
            <div>
              <h2 className="font-semibold text-white">
                What this result means
              </h2>
              <p className="mt-2 max-w-4xl text-sm leading-6 text-slate-400">
                Cyan marks pixels with unusually high local contrast. This
                transparent baseline can demonstrate segmentation and create
                measurable image features. It does not diagnose cancer, measure
                cell death, or prove a Tribonema treatment effect. Those claims
                require controlled treatment images and an independent viability
                assay.
              </p>
              <p className="mt-3 text-xs text-slate-500">
                Example image: LIVECell A172 · CC BY-NC 4.0 · Edlund et al.,
                2021
              </p>
            </div>
          </section>
        </div>
      </section>
    </main>
  );
}

function Control({
  label,
  output,
  min,
  max,
  step = 1,
  value,
  onChange,
}: {
  label: string;
  output: string;
  min: number;
  max: number;
  step?: number;
  value: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className="control-label">
      <span>
        {label}
        <output>{output}</output>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}

function Viewer({
  number,
  label,
  canvasRef,
  ariaLabel,
}: {
  number: string;
  label: string;
  canvasRef: React.RefObject<HTMLCanvasElement | null>;
  ariaLabel: string;
}) {
  return (
    <figure className="viewer">
      <figcaption>
        <span>{number}</span> {label}
      </figcaption>
      <div className="canvas-shell">
        <canvas ref={canvasRef} aria-label={ariaLabel} />
      </div>
    </figure>
  );
}

function Metric({
  label,
  value,
  unit,
}: {
  label: string;
  value: string;
  unit: string;
}) {
  return (
    <div className="lab-panel px-5 py-4">
      <p className="text-xs font-medium uppercase tracking-[0.08em] text-slate-500">
        {label}
      </p>
      <div className="mt-2 flex items-baseline gap-2">
        <strong className="font-mono text-2xl font-semibold text-white">
          {value}
        </strong>
        <span className="text-xs text-slate-500">{unit}</span>
      </div>
    </div>
  );
}
