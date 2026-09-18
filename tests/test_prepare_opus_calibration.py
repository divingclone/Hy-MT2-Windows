import sys
from pathlib import Path
import unittest
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from prepare_opus_calibration import ASSISTANT, BOS, EOT, USER, calibration_text, deduplicate, normalized, requests
from translate import translation_prompt


class OpusPreparationTests(unittest.TestCase):
    def test_cross_split_deduplication_in_both_languages(self):
        rows = [{'translation': {'en': 'A B', 'zh': '甲'}},
                {'translation': {'en': 'ａ  b', 'zh': '乙'}},
                {'translation': {'en': 'C', 'zh': '丙'}},
                {'translation': {'en': 'D', 'zh': '丁'}}]
        actual = deduplicate(rows, {('zh', normalized('丙'))})
        self.assertEqual([r['index'] for r in actual], [0, 3])

    def test_alignment_and_no_sentence_reuse(self):
        rows = [{'index': i, 'en': f'en {i}', 'zh': f'zh {i}'} for i in range(1000)]
        actual = requests(rows, 'test', 64, 23)
        self.assertEqual(actual, requests(rows, 'test', 64, 23))
        self.assertNotEqual(actual, requests(rows, 'test', 64, 24))
        used = [i for row in actual for i in row['sentence_indices']]
        self.assertEqual(len(used), len(set(used)))
        self.assertEqual({r['bucket'] for r in actual}, {'single', 'medium', 'long'})
        self.assertEqual({r['target_lang'] for r in actual}, {'English', 'Chinese'})
        self.assertEqual(Counter(r['bucket'] for r in actual), {'single': 32, 'medium': 16, 'long': 16})
        for bucket, expected in [('single', 16), ('medium', 8), ('long', 8)]:
            self.assertEqual(Counter(r['source_lang'] for r in actual if r['bucket'] == bucket),
                             {'en': expected, 'zh': expected})
        for row in actual:
            source, target = ('en', 'zh') if row['target_lang'] == 'Chinese' else ('zh', 'en')
            self.assertEqual(row['text'], '\n'.join(f'{source} {i}' for i in row['sentence_indices']))
            self.assertEqual(row['reference'], '\n'.join(f'{target} {i}' for i in row['sentence_indices']))

    def test_insufficient_pairs_fail_instead_of_repeating(self):
        with self.assertRaises(ValueError):
            requests([{'index': 0, 'en': 'A', 'zh': '甲'}], 'test', 2, 42)

    def test_non_multiple_counts_keep_each_bucket_balanced(self):
        rows = [{'index': i, 'en': f'en {i}', 'zh': f'zh {i}'} for i in range(1000)]
        for count in (1, 2, 3, 7, 13, 31, 65):
            actual = requests(rows, 'test', count, 42)
            for bucket in ('single', 'medium', 'long'):
                frequencies = Counter(row['source_lang'] for row in actual if row['bucket'] == bucket)
                self.assertLessEqual(abs(frequencies['en'] - frequencies['zh']), 1)

    def test_control_tokens_excluded_without_breaking_alignment(self):
        rows = [{'translation': {'en': 'good source', 'zh': '有效译文'}},
                {'translation': {'en': 'has ' + ASSISTANT, 'zh': 'another translation'}},
                {'translation': {'en': 'third source', 'zh': '<think>其他译文'}},
                {'translation': {'en': 'fourth source', 'zh': EOT}},
                {'translation': None}]
        self.assertEqual([row['index'] for row in deduplicate(rows)], [0])

    def test_calibration_uses_inference_prefix_reference_and_real_eog(self):
        row = {'text': 'Hello.', 'target_lang': 'Chinese', 'reference': '你好。'}
        expected = BOS + USER + translation_prompt('Hello.', 'Chinese') + ASSISTANT + '你好。' + EOT
        self.assertEqual(calibration_text([row]), expected)
        self.assertEqual(calibration_text([row, row]), expected + '\n' + expected)
        self.assertNotIn('<｜hy_place▁holder▁no▁8｜>', expected)
        with self.assertRaisesRegex(ValueError, 'control token'):
            calibration_text([{**row, 'reference': EOT}])


if __name__ == '__main__':
    unittest.main()
