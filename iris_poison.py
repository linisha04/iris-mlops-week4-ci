import argparse




def prepare_and_run_all(data_path: str, outdir: str, poisoning_rates: List[float], noise_std: float = 0.0):
data_path = Path(data_path)
outdir = Path(outdir)
outdir.mkdir(parents=True, exist_ok=True)


df = load_iris(str(data_path))


# Expect the species column to be named 'species' or last column
if 'species' in df.columns:
label_col = 'species'
else:
label_col = df.columns[-1]


X = df.drop(columns=[label_col])
y = df[label_col]


# Create a fixed clean validation set (20%) and keep it untouched
X_train_full, X_val, y_train_full, y_val = train_test_split(
X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
)


# Save validation set for reference
val_dir = outdir / "validation"
val_dir.mkdir(exist_ok=True)
pd.concat([X_val, y_val.reset_index(drop=True)], axis=1).to_csv(val_dir / "validation_set.csv", index=False)


experiment_name = "iris_poisoning_week4"
mlflow.set_tracking_uri(f"file:{str(outdir / 'mlruns')}")


results = []


for rate in poisoning_rates:
run_name = f"poison_{int(rate)}pct"
print(f"\n=== Running poisoning rate: {rate}% -> run: {run_name} ===")


Xp, yp, changed_idx = poison_training_set(X_train_full, y_train_full, rate=rate, noise_std=noise_std, seed=RANDOM_SEED)


run_outdir = outdir / run_name
run_outdir.mkdir(parents=True, exist_ok=True)


# Save poisoned training sample summary
pd.concat([Xp, yp.reset_index(drop=True)], axis=1).to_csv(run_outdir / "train_poisoned.csv", index=False)
with open(run_outdir / "poison_info.json", "w") as fh:
info = {"rate": rate, "n_changed": int(len(changed_idx)), "changed_indices": changed_idx.tolist()}
json.dump(info, fh, indent=2)


summary = train_and_log(Xp, yp, X_val, y_val, exp_name=experiment_name, run_name=run_name, outdir=run_outdir)


summary_record = {
"rate": rate,
"train_accuracy": summary["train_accuracy"],
"val_accuracy": summary["val_accuracy"],
"n_train": len(Xp),
"n_val": len(X_val)
}
results.append(summary_record)


# Save aggregated results
results_df = pd.DataFrame(results)
results_df.to_csv(outdir / "summary_results.csv", index=False)
print("\nAll runs finished. Summary saved to:", str(outdir / "summary_results.csv"))
print(results_df.to_string(index=False))




if __name__ == '__main__':
parser = argparse.ArgumentParser(description='Iris poisoning + training with MLflow')
parser.add_argument('--data', type=str, default='data/iris.csv', help='Path to iris.csv')
parser.add_argument('--outdir', type=str, default='artifacts/poison_runs', help='Output artifacts dir')
parser.add_argument('--rates', type=str, default='0,5,10,50', help='Comma-separated poisoning rates (percent)')
parser.add_argument('--noise', type=float, default=0.0, help='Gaussian noise stddev to add to poisoned rows')


args = parser.parse_args()


rates = [float(x.strip()) for x in args.rates.split(',') if x.strip()]


prepare_and_run_all(args.data, args.outdir, poisoning_rates=rates, noise_std=args.noise)
