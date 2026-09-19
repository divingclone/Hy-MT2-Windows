import test from 'node:test';
import assert from 'node:assert/strict';
import { cacheSaving } from '../src/cache-estimates.ts';
import { validateDraft, DEFAULT_SETTINGS, resetInference } from '../src/config.ts';

test('custom context keeps exact input and rejects blank, fraction and out of range', () => {
  const settings = { ...DEFAULT_SETTINGS, context: 1537 };
  assert.doesNotThrow(() => validateDraft(settings));
  assert.equal(settings.context, 1537);
  for (const context of [undefined, NaN, 0, 1537.5, 32769]) {
    assert.throws(() => validateDraft({ ...settings, context }));
  }
});

test('restore inference defaults preserves client connection and log settings', () => {
  const settings = { ...DEFAULT_SETTINGS, context: 8192, memoryPercent: 100, parallel: 128, gpu: 'GPU-custom', cache: 'f16', port: 19900, apiKeyEnabled: false, logMode: 'off' };
  const restored = resetInference(settings);
  assert.equal(restored.context, 2048);
  assert.equal(restored.memoryPercent, 30);
  assert.equal(restored.parallel, 0);
  assert.equal(restored.cache, 'q8_0');
  assert.equal(restored.gpu, '');
  assert.equal(restored.port, 19900);
  assert.equal(restored.apiKeyEnabled, false);
  assert.equal(restored.logMode, 'off');
  assert.equal(settings.context, 8192);
  for (const memoryPercent of [undefined, 9, 101, 30.5, NaN]) {
    assert.throws(() => validateDraft({ ...restored, memoryPercent }));
  }
});

test('cache savings include block scales and compare identical slot count, not total VRAM', () => {
  const plan = { parallel: 8, budget: { kv_elements_per_token_per_cache: 16384, kv_context_tokens_rounded: 2048, kv_total_mib: 544, estimated_total_mib: 4544 } };
  assert.deepEqual(cacheSaving(plan, 'q8_0'), { mib: 480, kvPercent: 46.875, totalPercent: 480 / 5024 * 100 });
  assert.deepEqual(cacheSaving(plan, 'q4_0'), { mib: 736, kvPercent: 71.875, totalPercent: 736 / 5024 * 100 });
  assert.equal(cacheSaving(null, 'q8_0'), null);
  assert.equal(cacheSaving(plan, 'f16').mib, 0);
});
