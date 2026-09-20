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
  const settings = { ...DEFAULT_SETTINGS, context: 8192, memoryPercent: 100, parallel: 128, gpu: 'GPU-custom', cache: 'bfloat16', port: 19900, apiKeyEnabled: false, logMode: 'off' };
  const restored = resetInference(settings);
  assert.equal(restored.context, 2048);
  assert.equal(restored.memoryPercent, 75);
  assert.equal(restored.parallel, 0);
  assert.equal(restored.cache, 'int8_per_token_head');
  assert.equal(restored.gpu, '');
  assert.equal(restored.port, 19900);
  assert.equal(restored.apiKeyEnabled, false);
  assert.equal(restored.logMode, 'off');
  assert.equal(settings.context, 8192);
  for (const memoryPercent of [undefined, 9, 101, 30.5, NaN]) {
    assert.throws(() => validateDraft({ ...restored, memoryPercent }));
  }
});

test('cache savings compare the same token pool including FP32 scales', () => {
 const plan = { parallel: 32, budget: { kv_token_capacity: 49152, kv_total_mib: 1584, estimated_total_mib: 5584 } };
 assert.deepEqual(cacheSaving(plan, 'int8_per_token_head'), { mib: 1488, kvPercent: 48.4375, totalPercent: 1488 / 7072 * 100 });
 assert.equal(cacheSaving(plan, 'bfloat16').mib, 0);
 assert.equal(cacheSaving(null, 'int8_per_token_head'), null);
});
