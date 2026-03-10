import numpy as np
from pathlib import Path
import json

base_path = Path('output_cosma')

models = ['lcdm', 'frhs', 'ndgp']

def main():
    for m in models:
        nsnaps = 27 if m == "lcdm" else 25

        path_1 = base_path / m / "seed_2080"
        path_2 = base_path / m / "seed_4257"

        if not path_1.is_dir():
            raise ValueError('path_1 not a directory')

        if not path_2.is_dir():
            raise ValueError('path_2 not a directory')

        print('computing', m)

        out_pk = base_path / m / "Pk_matter"
        out_pcurl = base_path / m / "Pk_curl"

        out_pk.mkdir(parents=True, exist_ok=True)
        out_pcurl.mkdir(parents=True, exist_ok=True)

        for i in range(nsnaps):
            path_snap1 = path_1 / f"snap_{i:03d}"
            path_snap2 = path_2 / f"snap_{i:03d}"

            pkm_1 = np.load(path_snap1/"Pk_m.npy")
            pkq_1 = np.load(path_snap1/"Pk_curl.npy")
            km_1 =  np.load(path_snap1/"k_m.npy")
            kq_1 =  np.load(path_snap1/"k_curl.npy")

            pkm_2 = np.load(path_snap2/"Pk_m.npy")
            pkq_2 = np.load(path_snap2/"Pk_curl.npy")
            km_2 =  np.load(path_snap2/"k_m.npy")
            kq_2 =  np.load(path_snap2/"k_curl.npy")

            # check if k's are equal
            np.testing.assert_array_equal(km_1,km_2)
            np.testing.assert_array_equal(kq_1,kq_2)

            Pk =    np.mean([pkm_1,pkm_2], axis=0)
            Pcurl = np.mean([pkq_1,pkq_2], axis=0)

            with open(path_snap1 / "snapshot_metadata.json") as f:
                meta = json.load(f)

            z = meta["redshift"]

            data_m = {
                "Pk": Pk,
                "k": km_1,
                "z": z
            }

            data_curl = {
                "Pcurl": Pcurl,
                "k": kq_1,
                "z": z
            }

            np.save(out_pcurl / f"{i:03d}.npy", data_curl)
            np.save(out_pk / f"{i:03d}.npy", data_m)
        
if __name__ == "__main__":
    main()
