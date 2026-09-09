'use client';

/**
 * The site's front page.
 *
 * It previously ran a box-filter threshold on an uploaded image and presented it
 * as a demonstration of the project. That was labelled honestly as a baseline,
 * but it was also the weakest algorithm the project contains -- 0.425 Dice
 * against 0.710 for labelling every pixel a cell -- and it was not the trained
 * model, so the demonstration showed the one method nobody should use.
 *
 * What runs here now is TriboGuard's real arithmetic, the same closed form the
 * Python package uses, verified equal by a parity test. No model weights, no
 * server: every number below follows from the chi-squared law of a sample
 * variance, which depends on the well count alone.
 */

import {
  FlaskConical,
  Microscope,
  ShieldAlert,
  ShieldCheck,
} from 'lucide-react';
import { useMemo, useState } from 'react';

import {
  type Assay,
  type Confidence,
  IMAGING,
  MTS,
  cost,
  relativeWidth,
  turnoverRatio,
  verdict,
  wellsConsumed,
  wellsForRatio,
} from '../lib/triboguard';

const PLATE_WELLS = 96;

/** The conclusions the 2022 Tribonema report drew, and what its design supports. */
const CLAIMS: ReadonlyArray<{ text: string; supported: boolean; why: string }> =
  [
    {
      text: 'Tribonema extract affected the cells',
      supported: true,
      why: 'An MTS drop against a matched untreated control.',
    },
    {
      text: 'The effect is comparable in size to doxorubicin',
      supported: true,
      why: 'Both measured on the same readout, on the same plate.',
    },
    {
      text: 'About 40% of the cells died',
      supported: false,
      why: 'MTS reports mitochondrial dehydrogenase activity, not cell number. A 40% fall is equally consistent with 40% of cells dying and with every cell surviving at 60% of its former rate.',
    },
    {
      text: 'The surviving 60% were resistant',
      supported: false,
      why: 'Needs cell counts, and a washout or re-treatment that was never run. A fraction is only resistant if it survives re-exposure.',
    },
    {
      text: 'The effect was killing rather than growth inhibition',
      supported: false,
      why: 'Needs cell counts and the spread between wells. The report states no replicate count anywhere.',
    },
    {
      text: 'It caused a specific death mechanism',
      supported: false,
      why: 'Apoptosis and necrosis differ by membrane integrity and caspase activity. Formalin fixation destroys that evidence before staining.',
    },
    {
      text: 'It is selective for cancer cells over normal cells',
      supported: false,
      why: 'The comparator is peritoneal lymphocytes — a different lineage, in primary culture. The contrast confounds cancer-versus-normal with macrophage-versus-lymphocyte.',
    },
  ];

const VERDICT_STYLE: Record<
  string,
  { chip: string; Icon: typeof ShieldCheck }
> = {
  none: {
    chip: 'bg-rose-500/15 text-rose-200 ring-rose-400/30',
    Icon: ShieldAlert,
  },
  no: {
    chip: 'bg-rose-500/15 text-rose-200 ring-rose-400/30',
    Icon: ShieldAlert,
  },
  borderline: {
    chip: 'bg-amber-500/15 text-amber-200 ring-amber-400/30',
    Icon: ShieldAlert,
  },
  yes: {
    chip: 'bg-emerald-500/15 text-emerald-200 ring-emerald-400/30',
    Icon: ShieldCheck,
  },
};

function Readout({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  hint?: string;
}) {
  return (
    <div className="rounded-lg border border-slate-700/40 bg-slate-950/40 px-4 py-3">
      <p className="eyebrow text-[0.62rem] text-slate-500">{label}</p>
      <p className="mt-1 font-mono text-xl font-semibold tabular-nums text-white">
        {value}
      </p>
      {hint ? (
        <p className="mt-0.5 text-[0.68rem] text-slate-500">{hint}</p>
      ) : null}
    </div>
  );
}

export default function Page() {
  // Opens on the design in the Tribonema report rather than an empty form, so
  // the first thing a visitor reads is the finding.
  const [wells, setWells] = useState(3);
  const [timepoints, setTimepoints] = useState(4);
  const [assay, setAssay] = useState<Assay>(MTS);
  const [confidence, setConfidence] = useState<Confidence>(95);

  const summary = useMemo(() => {
    const ratio = turnoverRatio(wells, confidence);
    return {
      ratio,
      width: relativeWidth(wells, confidence),
      consumed: wellsConsumed(wells, timepoints, assay),
      price: cost(wells, timepoints, assay),
      otherPrice: cost(wells, timepoints, assay.destructive ? IMAGING : MTS),
      needed: wellsForRatio(2, confidence),
      call: verdict(wells, confidence),
    };
  }, [wells, timepoints, assay, confidence]);

  const style = VERDICT_STYLE[summary.call.kind];
  const { Icon } = style;
  const overflowing = summary.consumed > PLATE_WELLS;

  return (
    <main className="min-h-screen bg-slate-950 text-slate-200">
      <header className="border-b border-slate-800/60 px-5 py-7 sm:px-8">
        <div className="mx-auto max-w-[1200px]">
          <p className="eyebrow">Experiment sufficiency</p>
          <h1 className="mt-1 text-3xl font-semibold tracking-tight text-white">
            TriboGuard
          </h1>
          <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-400">
            A treatment that halves a cell population may have killed half the
            cells or stopped half of them dividing. Those need different
            follow-up experiments, and a viability assay cannot tell them apart
            — the mean of a birth–death process depends on birth minus death, so
            the two produce identical average curves. This works out whether a
            design can separate them, and what the answer would cost.
          </p>
        </div>
      </header>

      <section className="mx-auto grid max-w-[1200px] gap-5 px-5 py-7 sm:px-8 lg:grid-cols-[300px_minmax(0,1fr)]">
        <aside className="lab-panel h-fit p-5 lg:sticky lg:top-6">
          <p className="eyebrow">Your design</p>

          <label className="mt-5 block">
            <span className="flex items-baseline justify-between text-xs font-medium uppercase tracking-wider text-slate-400">
              Wells per condition
              <span className="font-mono text-sm text-white">{wells}</span>
            </span>
            <input
              type="range"
              min={1}
              max={60}
              value={wells}
              onChange={(event) => setWells(Number(event.target.value))}
              className="mt-2 w-full accent-cyan-300"
            />
            <span className="mt-1 block text-[0.7rem] text-slate-500">
              The only control that moves the mechanism.
            </span>
          </label>

          <label className="mt-5 block">
            <span className="flex items-baseline justify-between text-xs font-medium uppercase tracking-wider text-slate-400">
              Timepoints
              <span className="font-mono text-sm text-white">{timepoints}</span>
            </span>
            <input
              type="range"
              min={2}
              max={12}
              value={timepoints}
              onChange={(event) => setTimepoints(Number(event.target.value))}
              className="mt-2 w-full accent-cyan-300"
            />
            <span className="mt-1 block text-[0.7rem] text-slate-500">
              Sharpens the growth curve. Buys nothing for the mechanism.
            </span>
          </label>

          <div className="mt-5">
            <span className="text-xs font-medium uppercase tracking-wider text-slate-400">
              Assay
            </span>
            <div className="mt-2 flex overflow-hidden rounded-lg border border-slate-700/60">
              {[MTS, IMAGING].map((option) => (
                <button
                  key={option.name}
                  type="button"
                  aria-pressed={assay.name === option.name}
                  onClick={() => setAssay(option)}
                  className={`flex-1 px-3 py-2 text-xs transition ${
                    assay.name === option.name
                      ? 'bg-cyan-400/20 text-cyan-100'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  {option.destructive ? 'MTS · endpoint' : 'Imaging · live'}
                </button>
              ))}
            </div>
            <span className="mt-1 block text-[0.7rem] text-slate-500">
              {assay.destructive
                ? 'MTS lyses the well it reads, so every timepoint needs its own wells.'
                : 'Imaging leaves the well alive, so the same wells carry every timepoint.'}
            </span>
          </div>

          <div className="mt-5">
            <span className="text-xs font-medium uppercase tracking-wider text-slate-400">
              Confidence
            </span>
            <div className="mt-2 flex overflow-hidden rounded-lg border border-slate-700/60">
              {([95, 90] as const).map((level) => (
                <button
                  key={level}
                  type="button"
                  aria-pressed={confidence === level}
                  onClick={() => setConfidence(level)}
                  className={`flex-1 px-3 py-2 text-xs transition ${
                    confidence === level
                      ? 'bg-cyan-400/20 text-cyan-100'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  {level}%
                </button>
              ))}
            </div>
          </div>
        </aside>

        <div className="flex flex-col gap-5">
          <div className="lab-panel p-6">
            <span
              className={`inline-flex items-center gap-2 rounded-md px-2.5 py-1 text-[0.68rem] font-semibold uppercase tracking-wider ring-1 ${style.chip}`}
            >
              <Icon size={14} aria-hidden="true" />
              {summary.call.label}
            </span>
            <p className="mt-4 text-xl leading-relaxed text-white">
              {summary.call.sentence}
            </p>

            <div className="mt-5 grid gap-3 sm:grid-cols-3">
              <Readout
                label="Split known to within"
                value={
                  Number.isFinite(summary.ratio)
                    ? `×${summary.ratio < 10 ? summary.ratio.toFixed(1) : Math.round(summary.ratio).toLocaleString()}`
                    : 'nothing'
                }
                hint="ratio of the interval's ends"
              />
              <Readout
                label="Relative width"
                value={
                  Number.isFinite(summary.width)
                    ? summary.width.toFixed(2)
                    : '—'
                }
                hint="its span over the estimate"
              />
              <Readout
                label="Wells for a factor of 2"
                value={summary.needed === null ? '—' : String(summary.needed)}
                hint="per condition"
              />
            </div>

            <p className="mt-4 text-xs text-slate-500">
              Costs {summary.price.toFixed(2)} in wells and reads. The same
              information under{' '}
              {assay.destructive
                ? 'live imaging'
                : 'an endpoint assay like MTS'}{' '}
              costs {summary.otherPrice.toFixed(2)}.
            </p>
          </div>

          <div className="lab-panel p-6">
            <p className="eyebrow">What the plate gives up</p>
            <div className="mt-4 flex flex-wrap items-start gap-6">
              <div
                className="grid w-full max-w-[380px] grid-cols-12 gap-[3px]"
                aria-hidden="true"
              >
                {Array.from({ length: PLATE_WELLS }, (_, index) => (
                  <span
                    key={index}
                    className={`aspect-square rounded-full border ${
                      index < summary.consumed
                        ? overflowing
                          ? 'border-rose-400/60 bg-rose-500/70'
                          : 'border-cyan-300/60 bg-cyan-400/70'
                        : 'border-slate-700/50 bg-slate-800/40'
                    }`}
                  />
                ))}
              </div>
              <p className="min-w-[220px] flex-1 text-sm leading-6 text-slate-400">
                <strong className="text-slate-200">
                  {summary.consumed} of {PLATE_WELLS} wells
                </strong>{' '}
                for one condition.{' '}
                {assay.destructive
                  ? 'An endpoint assay cannot read a well twice, so every timepoint needs its own set. A dose series and its controls multiply this again.'
                  : 'Live imaging reads the same wells at every timepoint, so timepoints are free in plate space.'}
                {overflowing ? (
                  <strong className="mt-2 block text-rose-300">
                    This design no longer fits one plate.
                  </strong>
                ) : null}
              </p>
            </div>
          </div>

          <div className="lab-panel p-6">
            <p className="eyebrow">
              Case study · Tribonema extract on RAW264.7, 2022
            </p>
            <ul className="mt-4 flex flex-col gap-3">
              {CLAIMS.map((claim) => (
                <li
                  key={claim.text}
                  className="flex flex-col gap-1 border-t border-slate-800/60 pt-3 first:border-0 first:pt-0"
                >
                  <div className="flex flex-wrap items-baseline gap-2">
                    <span
                      className={`rounded px-2 py-0.5 font-mono text-[0.62rem] uppercase tracking-wider ${
                        claim.supported
                          ? 'bg-emerald-500/15 text-emerald-200'
                          : 'bg-rose-500/15 text-rose-200'
                      }`}
                    >
                      {claim.supported ? 'supported' : 'not established'}
                    </span>
                    <span className="text-sm text-slate-200">{claim.text}</span>
                  </div>
                  <p className="text-xs leading-5 text-slate-500">
                    {claim.why}
                  </p>
                </li>
              ))}
            </ul>
            <p className="mt-4 text-xs leading-5 text-slate-500">
              Two of seven conclusions survive, and the reported effect is one
              of them. This is a statement about which further conclusions the
              measurements can carry — not about whether the extract works.
            </p>
          </div>

          <div className="lab-panel flex flex-wrap items-start gap-4 p-6">
            <FlaskConical
              className="text-cyan-200"
              size={20}
              aria-hidden="true"
            />
            <div className="min-w-[240px] flex-1">
              <h2 className="font-semibold text-white">
                What this page is not
              </h2>
              <p className="mt-2 text-sm leading-6 text-slate-400">
                It does not diagnose anything, analyse an image, or demonstrate
                that Tribonema works. It computes what a given experimental
                design can and cannot establish about a mechanism. Every figure
                comes from the chi-squared law of a sample variance and runs
                entirely in this browser, using the same tabulated quantiles as
                the Python package so the two agree exactly.
              </p>
              <p className="mt-3 flex items-center gap-2 text-xs text-slate-500">
                <Microscope size={13} aria-hidden="true" />
                Cell-segmentation results live in the repository&apos;s
                generated report; this page deliberately does not re-implement
                them.
              </p>
            </div>
          </div>
        </div>
      </section>
    </main>
  );
}
