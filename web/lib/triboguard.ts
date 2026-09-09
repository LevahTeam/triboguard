/**
 * TriboGuard's arithmetic, in the browser.
 *
 * The site used to run a box-filter threshold and present it as a demonstration
 * of the project. It was honest about being a baseline, but the baseline is the
 * weakest thing the project contains -- measured at 0.425 Dice against 0.710 for
 * labelling every pixel a cell -- so the demonstration showed the one algorithm
 * nobody should use.
 *
 * What runs here instead is the real thing. Every number TriboGuard reports
 * about a design comes from the chi-squared law of a sample variance, which is a
 * function of the well count alone, so it needs no model weights and no server.
 * The quantiles are tabulated from scipy rather than approximated, which is what
 * lets a parity test assert the browser and the Python agree exactly.
 */

import { C90_HI, C90_LO, C95_HI, C95_LO } from './chi2';

export type Confidence = 90 | 95;

const TABLES: Record<
  Confidence,
  readonly [readonly number[], readonly number[]]
> = {
  95: [C95_LO, C95_HI],
  90: [C90_LO, C90_HI],
};

/** Below two wells there is no spread to measure, and no mechanism at any price. */
export const MINIMUM_USEFUL_WELLS = 2;

/** A design needing more wells than this is not one a laboratory will run. */
export const MAXIMUM_SEARCHED_WELLS = 400;

function degreesOfFreedom(wells: number, confidence: Confidence) {
  const [lo, hi] = TABLES[confidence];
  const df = wells - 1;
  if (df < 1 || df > lo.length) return null;
  return { df, lo: lo[df - 1], hi: hi[df - 1] };
}

/**
 * Width of the turnover interval as a multiple of the estimate.
 *
 * Not the same as {@link turnoverRatio}, and the two were conflated three times
 * during development. This one is the interval's span divided by the estimate.
 */
export function relativeWidth(
  wells: number,
  confidence: Confidence = 95,
): number {
  const d = degreesOfFreedom(wells, confidence);
  if (!d) return wells < MINIMUM_USEFUL_WELLS ? Infinity : 0;
  return d.df / d.lo - d.df / d.hi;
}

/**
 * How many times wider the interval's top is than its bottom.
 *
 * This is what "known to within a factor of X" actually names. At three wells it
 * is 146 where the relative width is 39, so the labels matter.
 */
export function turnoverRatio(
  wells: number,
  confidence: Confidence = 95,
): number {
  const d = degreesOfFreedom(wells, confidence);
  if (!d) return wells < MINIMUM_USEFUL_WELLS ? Infinity : 1;
  return d.hi / d.lo;
}

/** Fewest wells whose interval spans no more than `target`-fold, or null. */
export function wellsForRatio(
  target: number,
  confidence: Confidence = 95,
): number | null {
  if (target <= 1) return null;
  for (
    let wells = MINIMUM_USEFUL_WELLS;
    wells <= MAXIMUM_SEARCHED_WELLS;
    wells += 1
  ) {
    if (turnoverRatio(wells, confidence) <= target) return wells;
  }
  return null;
}

export type Assay = {
  readonly name: string;
  readonly perWell: number;
  readonly perMeasurement: number;
  /** True for an endpoint assay such as MTS, where reading the well destroys it. */
  readonly destructive: boolean;
};

export const MTS: Assay = {
  name: 'MTS',
  perWell: 0.8,
  perMeasurement: 0.35,
  destructive: true,
};

export const IMAGING: Assay = {
  name: 'live-cell imaging',
  perWell: 0.8,
  perMeasurement: 0.2,
  destructive: false,
};

/**
 * Wells the plate actually gives up.
 *
 * A destructive assay cannot re-read a well, so every timepoint needs its own
 * set. Counting cost per measurement alone hides this entirely.
 */
export function wellsConsumed(
  wells: number,
  timepoints: number,
  assay: Assay,
): number {
  return assay.destructive ? wells * timepoints : wells;
}

export function cost(wells: number, timepoints: number, assay: Assay): number {
  return (
    wellsConsumed(wells, timepoints, assay) * assay.perWell +
    wells * timepoints * assay.perMeasurement
  );
}

export type Verdict = {
  readonly kind: 'none' | 'no' | 'borderline' | 'yes';
  readonly label: string;
  readonly sentence: string;
};

/** Whether this design can separate cell killing from growth inhibition. */
export function verdict(wells: number, confidence: Confidence = 95): Verdict {
  if (wells < MINIMUM_USEFUL_WELLS) {
    return {
      kind: 'none',
      label: 'Cannot separate',
      sentence:
        'With one well per condition there is no spread between wells, so killing and growth inhibition are indistinguishable at any number of timepoints.',
    };
  }
  const ratio = turnoverRatio(wells, confidence);
  const width = relativeWidth(wells, confidence);
  const shown =
    ratio < 10 ? ratio.toFixed(1) : Math.round(ratio).toLocaleString();
  if (width > 5) {
    return {
      kind: 'no',
      label: 'Cannot separate',
      sentence: `The birth-to-death split is pinned only to within a factor of ${shown}. A single-mechanism conclusion is not supported by this design.`,
    };
  }
  if (width > 1.5) {
    return {
      kind: 'borderline',
      label: 'Borderline',
      sentence: `The split is pinned to within a factor of ${shown} — enough to rule out extremes, not enough to name one mechanism with confidence.`,
    };
  }
  return {
    kind: 'yes',
    label: 'Can separate',
    sentence: `The split is pinned to within a factor of ${shown}. This design can distinguish cell killing from growth inhibition.`,
  };
}
