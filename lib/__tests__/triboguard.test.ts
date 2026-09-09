/**
 * The browser must agree with the Python package, not merely be close to it.
 *
 * The site reimplements TriboGuard's arithmetic so it can run without a server.
 * A reimplementation that drifts is worse than no demo, because both halves keep
 * working and quietly disagree. Every expected value below is generated from the
 * Python package by scripts/generate_parity_fixtures.py.
 */

import { describe, expect, it } from 'vitest';

import fixtures from '../__fixtures__/python-parity.json';
import {
  type Assay,
  type Confidence,
  IMAGING,
  MINIMUM_USEFUL_WELLS,
  MTS,
  cost,
  relativeWidth,
  turnoverRatio,
  verdict,
  wellsConsumed,
  wellsForRatio,
} from '../triboguard';

/** Agreement that means the same thing at 0.09 and at 1018. */
function relativeError(got: number, expected: number): number {
  return Math.abs(got - expected) / Math.abs(expected);
}

describe('parity with the Python package', () => {
  it.each(fixtures.intervals)(
    'matches relative width at $wells wells, $confidence%',
    ({ wells, confidence, relativeWidth: expected }) => {
      const got = relativeWidth(wells, confidence as Confidence);
      if (expected === null) {
        expect(got).toBe(Infinity);
        return;
      }
      // Relative, not absolute. toBeCloseTo compares an absolute difference,
      // which is meaninglessly strict at 1018 and meaninglessly loose at 0.09;
      // the table is exact to double precision, so the agreement to assert is
      // relative.
      expect(relativeError(got, expected)).toBeLessThan(1e-12);
    },
  );

  it.each(fixtures.intervals)(
    'matches the turnover ratio at $wells wells, $confidence%',
    ({ wells, confidence, turnoverRatio: expected }) => {
      const got = turnoverRatio(wells, confidence as Confidence);
      if (expected === null) {
        expect(got).toBe(Infinity);
        return;
      }
      expect(relativeError(got, expected)).toBeLessThan(1e-12);
    },
  );

  it.each(fixtures.wellsForRatio)(
    'needs $wells wells to reach a factor of $target',
    ({ target, confidence, wells }) => {
      expect(wellsForRatio(target, confidence as Confidence)).toBe(wells);
    },
  );

  it.each(fixtures.costs)(
    'prices $wells wells x $timepoints timepoints (destructive: $destructive)',
    ({
      wells,
      timepoints,
      destructive,
      wellsConsumed: consumed,
      cost: price,
    }) => {
      const assay: Assay = destructive ? MTS : IMAGING;
      expect(wellsConsumed(wells, timepoints, assay)).toBe(consumed);
      expect(cost(wells, timepoints, assay)).toBeCloseTo(price, 10);
    },
  );
});

describe('the two uncertainty measures stay distinct', () => {
  it('the ratio is much larger than the width at three wells', () => {
    // 146 against 39. They were conflated three times during development, so
    // this asserts they are not the same number.
    expect(turnoverRatio(3)).toBeGreaterThan(3 * relativeWidth(3));
  });

  it('both fall monotonically as wells are added', () => {
    const wells = [3, 6, 12, 24, 48];
    const ratios = wells.map((w) => turnoverRatio(w));
    const widths = wells.map((w) => relativeWidth(w));
    expect(ratios).toEqual([...ratios].sort((a, b) => b - a));
    expect(widths).toEqual([...widths].sort((a, b) => b - a));
  });
});

describe('a design with no spread', () => {
  it('reports one well as unbounded rather than as a number', () => {
    expect(relativeWidth(1)).toBe(Infinity);
    expect(turnoverRatio(1)).toBe(Infinity);
  });

  it('refuses to name a mechanism', () => {
    expect(verdict(MINIMUM_USEFUL_WELLS - 1).kind).toBe('none');
  });
});

describe('the destructive assay costs the plate, not the information', () => {
  it('consumes a well per timepoint', () => {
    expect(wellsConsumed(3, 4, MTS)).toBe(12);
    expect(wellsConsumed(3, 4, IMAGING)).toBe(3);
  });

  it('charges more for the same design', () => {
    expect(cost(3, 4, MTS)).toBeGreaterThan(cost(3, 4, IMAGING));
  });

  it('reports the same uncertainty either way, because the assay does not change it', () => {
    // The estimator reads a mean and a variance per timepoint and never follows
    // one well through time, so an endpoint assay loses money and not evidence.
    expect(relativeWidth(3)).toBe(relativeWidth(3));
    expect(verdict(3).sentence).toBe(verdict(3).sentence);
  });
});

describe('the verdict tracks the evidence', () => {
  it('refuses at the design in the Tribonema report', () => {
    expect(verdict(3).kind).toBe('no');
  });

  it('becomes decisive with enough wells', () => {
    expect(verdict(60).kind).toBe('yes');
  });

  it('never claims more than the interval allows', () => {
    const kinds = [3, 12, 24, 40, 60].map((w) => verdict(w).kind);
    const rank = { none: 0, no: 1, borderline: 2, yes: 3 } as const;
    const ranks = kinds.map((k) => rank[k]);
    expect(ranks).toEqual([...ranks].sort((a, b) => a - b));
  });
});
