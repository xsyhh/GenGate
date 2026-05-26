import json
import subprocess
import tempfile
import unittest
from pathlib import Path


class PrepareMathDataTest(unittest.TestCase):
    def test_end_to_end_conversion_and_unique_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            # Mock MATH-500
            math500_dir = root / "data" / "MATH-500"
            math500_dir.mkdir(parents=True)
            with (math500_dir / "test.jsonl").open("w", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "problem": "p1",
                            "solution": "... \\boxed{42}",
                            "answer": "42",
                            "subject": "Algebra",
                            "level": 1,
                            "unique_id": "test/algebra/1.json",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            # Mock hendrycks parquet
            hm_dir = root / "data" / "hendrycks_math" / "algebra"
            hm_dir.mkdir(parents=True)

            # Write parquet via pandas (openai env has pandas+pyarrow in target usage)
            train_pq = str(hm_dir / "train-00000-of-00001.parquet")
            test_pq = str(hm_dir / "test-00000-of-00001.parquet")
            code = (
                "import pandas as pd; "
                "df=pd.DataFrame(["
                "{'problem':'h_train','level':'Level 2','type':'Algebra','solution':'x=\\\\boxed{7}'},"
                "{'problem':'h_test','level':'Level 3','type':'Algebra','solution':'ans \\\\boxed{9}'}"
                "]);"
                f"df.iloc[[0]].to_parquet(r'{train_pq}', index=False);"
                f"df.iloc[[1]].to_parquet(r'{test_pq}', index=False)"
            )
            subprocess.run(["python", "-c", code], check=True)

            out_dir = root / "out"
            out_dir.mkdir()

            script = Path(__file__).resolve().parents[1] / "prepare_math_data.py"
            subprocess.run(
                [
                    "python",
                    str(script),
                    "--math500-dir",
                    str(math500_dir),
                    "--hendrycks-dir",
                    str(root / "data" / "hendrycks_math"),
                    "--output-dir",
                    str(out_dir),
                ],
                check=True,
            )

            hm_train = [
                json.loads(x)
                for x in (out_dir / "hendrycks_math_train.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            hm_test = [
                json.loads(x)
                for x in (out_dir / "hendrycks_math_test.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            m500 = [
                json.loads(x)
                for x in (out_dir / "math500_test.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]

            self.assertEqual(hm_train[0]["answer"], "7")
            self.assertEqual(hm_test[0]["answer"], "9")

            # Ensure required fields
            required = {"problem", "solution", "subject", "answer", "level", "unique_id"}
            for row in hm_train + hm_test + m500:
                self.assertTrue(required.issubset(row.keys()))

            # Ensure unique_id global disjointness
            all_ids = [r["unique_id"] for r in hm_train + hm_test + m500]
            self.assertEqual(len(all_ids), len(set(all_ids)))


if __name__ == "__main__":
    unittest.main()
