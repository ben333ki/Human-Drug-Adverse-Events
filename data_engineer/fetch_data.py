# fetch_data.py
import requests
import pandas as pd
import time
from config import OPENFDA_URL, LIMIT, GLP1_DRUGS, RAW_DATA_PATH, API_KEY

def fetch_openfda_multi_drugs():
    all_records = []

    for drug in GLP1_DRUGS:
        print(f"Fetching data for {drug} ...")
        skip = 0
        while True:
            search_query = f'patient.drug.medicinalproduct:"{drug}"'
            url = f"{OPENFDA_URL}?api_key={API_KEY}&limit={LIMIT}&skip={skip}&search={search_query}"
            resp = requests.get(url)
            if resp.status_code != 200:
                print(f"Finished {drug} or error: {resp.status_code}")
                break

            data = resp.json()
            results = data.get("results", [])
            if not results:
                break

            for r in results:
                patient = r.get("patient", {})
                drugs = patient.get("drug", [])
                reactions = patient.get("reaction", [])
                for d in drugs:
                    drug_name = d.get("medicinalproduct")
                    if drug_name not in GLP1_DRUGS:
                        continue
                    for rxn in reactions:
                        all_records.append({
                            "patient_age": patient.get("patientonsetage"),
                            "age_unit": patient.get("patientonsetageunit"),
                            "sex": patient.get("patientsex"),
                            "drug_name": drug_name,
                            "drug_route": d.get("drugadministrationroute"),
                            "reaction": rxn.get("reactionmeddrapt"),
                            "seriousness": r.get("serious"),
                            "receivedate": r.get("receivedate")
                        })

            skip += LIMIT
            time.sleep(0.1)  # ลดโอกาส hit rate limit

    # Save as CSV
    df = pd.DataFrame(all_records)
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"Total records fetched: {len(df)} and saved to {RAW_DATA_PATH}")
    return df

if __name__ == "__main__":
    fetch_openfda_multi_drugs()
