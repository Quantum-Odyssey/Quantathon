from __future__ import annotations

import time
from dataclasses import dataclass
from math import ceil, log2
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from matplotlib.backends.backend_pdf import PdfPages
from qiskit import transpile
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.circuit.library import (
    efficient_su2,
    pauli_feature_map,
    z_feature_map,
    zz_feature_map,
)
from qiskit.quantum_info import Statevector
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC


sns.set_theme(style="white", context="notebook")


@dataclass
class KernelBundle:
    train: np.ndarray
    test: np.ndarray
    circuit: QuantumCircuit
    generation_seconds: float


class QSVMExperimentAnalysis:
    """Análisis reproducible basado en la preparación de datos de qsvm.ipynb."""

    def __init__(self, repo_root: Path, output_dir: Path, random_seed: int = 42):
        self.repo_root = Path(repo_root)
        self.data_path = self.repo_root / "data" / "water_potability.csv"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.random_seed = random_seed

    @staticmethod
    def slug(config: dict) -> str:
        return f"n_{config['N']}_dim_{config['dim']}_{config['featuremap']}"

    def prepare_data(self, n: int, dim: int, scaling: str = "standard_pca"):
        data = pd.read_csv(self.data_path).dropna()
        data = data.sample(frac=1, random_state=self.random_seed)
        potable = data.loc[data["Potability"] == 1]
        non_potable = data.loc[data["Potability"] == 0]
        samples_per_class = int((n / 0.8) / 2)
        data = pd.concat(
            [potable.iloc[:samples_per_class], non_potable.iloc[:samples_per_class]]
        ).sample(frac=1, random_state=self.random_seed)

        x = data.drop(columns="Potability")
        y = data["Potability"]
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=self.random_seed
        )

        standard = StandardScaler()
        x_train = standard.fit_transform(x_train)
        x_test = standard.transform(x_test)
        if dim != x_train.shape[1]:
            pca = PCA(n_components=dim, random_state=self.random_seed)
            x_train = pca.fit_transform(x_train)
            x_test = pca.transform(x_test)

        if scaling == "minmax_0_pi":
            scaler = MinMaxScaler(feature_range=(0, np.pi))
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)
        elif scaling == "minmax_minus_pi_pi":
            scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)
        elif scaling != "standard_pca":
            raise ValueError(f"Escalado no soportado: {scaling}")

        if len(x_train) != n:
            raise ValueError(f"N={n} produjo {len(x_train)} muestras de entrenamiento.")
        return x_train, x_test, y_train.to_numpy(), y_test.to_numpy()

    @staticmethod
    def _entangling_edges(num_qubits: int, topology: str):
        if num_qubits < 2:
            return []
        if topology == "linear":
            return [(i, i + 1) for i in range(num_qubits - 1)]
        if topology in {"circular", "ring"}:
            return [(i, (i + 1) % num_qubits) for i in range(num_qubits)]
        if topology == "full":
            return [(i, j) for i in range(num_qubits) for j in range(i + 1, num_qubits)]
        raise ValueError(f"Topología no soportada: {topology}")

    @classmethod
    def custom_feature_map(
        cls, num_qubits: int, beta: float = 1.0, reps: int = 1, topology: str = "circular"
    ):
        x = ParameterVector("x", num_qubits * reps)
        circuit = QuantumCircuit(num_qubits, name="CustomQKE")
        parameter = 0
        for _ in range(reps):
            for qubit in range(num_qubits):
                circuit.h(qubit)
                circuit.ry(beta * x[parameter], qubit)
                parameter += 1
            if num_qubits > 1:
                circuit.t(1)
            circuit.barrier()
            for control, target in cls._entangling_edges(num_qubits, topology):
                circuit.cz(control, target)
        return circuit

    @staticmethod
    def _repeat_features(x: np.ndarray, parameter_count: int):
        repeats = ceil(parameter_count / x.shape[1])
        return np.tile(x, (1, repeats))[:, :parameter_count]

    def build_feature_map(
        self,
        name: str,
        x_train: np.ndarray,
        x_test: np.ndarray,
        reps: int = 1,
        entanglement: str = "circular",
        paulis: Iterable[str] | None = None,
    ):
        dimension = x_train.shape[1]
        if name == "efficient_su2":
            num_qubits = max(1, ceil(log2(dimension)) - 1)
            circuit = efficient_su2(
                num_qubits=num_qubits,
                reps=reps,
                entanglement=entanglement,
                insert_barriers=True,
            )
        elif name == "pauli_feature_map":
            circuit = pauli_feature_map(
                feature_dimension=dimension,
                reps=reps,
                paulis=list(paulis or ["Y", "ZZ"]),
                entanglement=entanglement,
            )
        elif name == "custom_q_kernel":
            circuit = self.custom_feature_map(
                dimension, beta=1.0, reps=reps, topology=entanglement
            )
        elif name == "z_feature_map":
            circuit = z_feature_map(feature_dimension=dimension, reps=reps)
        elif name == "zz_feature_map":
            circuit = zz_feature_map(
                feature_dimension=dimension, reps=reps, entanglement=entanglement
            )
        else:
            raise ValueError(f"Feature map no soportado: {name}")

        x_train = self._repeat_features(x_train, circuit.num_parameters)
        x_test = self._repeat_features(x_test, circuit.num_parameters)
        return circuit, x_train, x_test

    @staticmethod
    def exact_kernel(circuit: QuantumCircuit, x_train: np.ndarray, x_test: np.ndarray):
        started = time.perf_counter()
        train_states = np.vstack(
            [Statevector.from_instruction(circuit.assign_parameters(row)).data for row in x_train]
        )
        test_states = np.vstack(
            [Statevector.from_instruction(circuit.assign_parameters(row)).data for row in x_test]
        )
        train_kernel = np.abs(train_states.conj() @ train_states.T) ** 2
        test_kernel = np.abs(test_states.conj() @ train_states.T) ** 2
        train_kernel = (train_kernel + train_kernel.T) / 2
        np.fill_diagonal(train_kernel, 1.0)
        return KernelBundle(
            train=train_kernel,
            test=test_kernel,
            circuit=circuit,
            generation_seconds=time.perf_counter() - started,
        )

    def generate_bundle(self, config: dict, **variant):
        x_train, x_test, y_train, y_test = self.prepare_data(
            config["N"], config["dim"], variant.get("scaling", "standard_pca")
        )
        circuit, x_train, x_test = self.build_feature_map(
            config["featuremap"],
            x_train,
            x_test,
            reps=variant.get("reps", 1),
            entanglement=variant.get("entanglement", "circular"),
            paulis=variant.get("paulis"),
        )
        return self.exact_kernel(circuit, x_train, x_test), y_train, y_test

    @staticmethod
    def classification_metrics(bundle: KernelBundle, y_train, y_test):
        started = time.perf_counter()
        model = SVC(kernel="precomputed")
        model.fit(bundle.train, y_train)
        predictions = model.predict(bundle.test)
        svc_seconds = time.perf_counter() - started
        return {
            "accuracy": accuracy_score(y_test, predictions),
            "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
            "f1": f1_score(y_test, predictions, zero_division=0),
            "svc_seconds": svc_seconds,
        }

    @staticmethod
    def cross_validation(kernel: np.ndarray, labels: np.ndarray, folds: int = 5):
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        scores = []
        for train_index, validation_index in splitter.split(kernel, labels):
            model = SVC(kernel="precomputed")
            model.fit(kernel[np.ix_(train_index, train_index)], labels[train_index])
            prediction = model.predict(kernel[np.ix_(validation_index, train_index)])
            scores.append(balanced_accuracy_score(labels[validation_index], prediction))
        return float(np.mean(scores)), float(np.std(scores))

    @staticmethod
    def target_alignment(kernel: np.ndarray, labels: np.ndarray):
        signed = np.where(labels == 1, 1.0, -1.0)
        target = np.outer(signed, signed)
        denominator = np.linalg.norm(kernel, "fro") * np.linalg.norm(target, "fro")
        return float(np.sum(kernel * target) / denominator) if denominator else np.nan

    @staticmethod
    def similarities(kernel: np.ndarray, labels: np.ndarray):
        same = labels[:, None] == labels[None, :]
        diagonal = np.eye(len(labels), dtype=bool)
        intra = kernel[same & ~diagonal]
        inter = kernel[~same]
        return intra, inter

    @staticmethod
    def spectrum(kernel: np.ndarray):
        eigenvalues = np.linalg.eigvalsh((kernel + kernel.T) / 2)[::-1]
        positive = np.clip(eigenvalues, 0, None)
        probabilities = positive / positive.sum() if positive.sum() else positive
        nonzero = probabilities[probabilities > 0]
        effective_rank = float(np.exp(-np.sum(nonzero * np.log(nonzero))))
        numerical_rank = int(np.linalg.matrix_rank(kernel, tol=1e-8))
        return eigenvalues, effective_rank, numerical_rank

    @staticmethod
    def circuit_cost(circuit: QuantumCircuit):
        compiled = transpile(
            circuit,
            basis_gates=["rz", "sx", "x", "cx"],
            optimization_level=3,
            seed_transpiler=42,
        )
        two_qubit = sum(
            1 for instruction in compiled.data if len(instruction.qubits) == 2
        )
        return compiled, int(compiled.depth()), int(two_qubit), int(compiled.size())

    def save_circuits(
        self,
        slug: str,
        circuit: QuantumCircuit,
        compiled: QuantumCircuit,
        show: bool = False,
    ):
        for suffix, item in (("circuito", circuit), ("circuito_transpilado", compiled)):
            figure = item.draw(output="mpl", fold=24, idle_wires=False)
            figure.savefig(self.output_dir / f"{slug}_{suffix}.pdf", bbox_inches="tight")
            if show:
                print(f"\nImagen: {slug}_{suffix}")
                display(figure)
            plt.close(figure)

    def save_table(self, frame: pd.DataFrame, filename: str, show: bool = False):
        aliases = {
            "balanced_accuracy": "bal_accuracy",
            "svc_seconds": "svc_s",
            "cv_balanced_accuracy_mean": "cv_bal_mean",
            "cv_balanced_accuracy_std": "cv_bal_std",
            "kernel_target_alignment": "alineación",
            "intra_class_similarity_mean": "sim_intra",
            "inter_class_similarity_mean": "sim_inter",
            "effective_rank": "rango_efectivo",
            "numerical_rank": "rango_numérico",
            "circuit_depth": "profundidad",
            "two_qubit_gates": "compuertas_2q",
            "gate_count": "compuertas_total",
            "kernel_generation_seconds": "kernel_s",
            "train_kernel_circuits": "circuitos_train",
            "test_kernel_circuits": "circuitos_test",
            "depolarizing_probability": "prob_ruido",
        }
        frame = frame.rename(columns=aliases)

        def formatted(source):
            result = source.copy()
            for column in result.select_dtypes(include=[np.number]).columns:
                result[column] = result[column].map(lambda value: f"{value:.5g}")
            return result

        def add_page(pdf, source, width=None, height=None):
            source = formatted(source)
            width = width or max(7, 1.35 * len(source.columns))
            height = height or max(1.8, 0.42 * (len(source) + 1))
            figure, axis = plt.subplots(figsize=(width, height))
            axis.axis("off")
            table = axis.table(
                cellText=source.values,
                colLabels=source.columns,
                loc="center",
                cellLoc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.25)
            pdf.savefig(figure, bbox_inches="tight")
            if show:
                print(f"\nImagen: {Path(filename).stem}")
                display(figure)
            plt.close(figure)

        path = self.output_dir / filename
        with PdfPages(path) as pdf:
            if len(frame) == 1 and len(frame.columns) > 8:
                vertical = frame.T.reset_index()
                vertical.columns = ["métrica", "valor"]
                vertical["valor"] = vertical["valor"].map(
                    lambda value: f"{value:.5g}"
                    if isinstance(value, (int, float, np.number))
                    else str(value)
                )
                add_page(pdf, vertical, width=8, height=max(5, 0.34 * len(vertical)))
            elif len(frame.columns) > 8:
                identifiers = [
                    column for column in ("N", "dim", "featuremap") if column in frame.columns
                ]
                values = [column for column in frame.columns if column not in identifiers]
                for start in range(0, len(values), 4):
                    add_page(pdf, frame[identifiers + values[start : start + 4]], width=12)
            else:
                add_page(pdf, frame)

    def save_base_plots(
        self, slug: str, bundle: KernelBundle, labels: np.ndarray, eigenvalues: np.ndarray
    ):
        figure, axis = plt.subplots(figsize=(6, 5))
        sns.heatmap(bundle.train, cmap="viridis", vmin=0, vmax=1, square=True, ax=axis)
        axis.set_xlabel("Muestra")
        axis.set_ylabel("Muestra")
        figure.savefig(self.output_dir / f"{slug}_kernel_mapa_calor.pdf", bbox_inches="tight")
        plt.close(figure)

        upper = bundle.train[np.triu_indices_from(bundle.train, k=1)]
        figure, axis = plt.subplots(figsize=(6, 4))
        sns.histplot(upper, bins=25, kde=True, ax=axis)
        axis.set_xlabel("Similitud")
        axis.set_ylabel("Frecuencia")
        figure.savefig(self.output_dir / f"{slug}_kernel_distribucion.pdf", bbox_inches="tight")
        plt.close(figure)

        intra, inter = self.similarities(bundle.train, labels)
        figure, axis = plt.subplots(figsize=(6, 4))
        sns.kdeplot(intra, label="Intraclase", fill=True, alpha=0.3, ax=axis)
        sns.kdeplot(inter, label="Interclase", fill=True, alpha=0.3, ax=axis)
        axis.set_xlabel("Similitud")
        axis.set_ylabel("Densidad")
        axis.legend()
        figure.savefig(self.output_dir / f"{slug}_similitud_intra_interclase.pdf", bbox_inches="tight")
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(6, 4))
        axis.plot(np.arange(1, len(eigenvalues) + 1), eigenvalues, marker="o", markersize=3)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xlabel("Índice")
        axis.set_ylabel("Valor propio")
        figure.savefig(self.output_dir / f"{slug}_espectro_valores_propios.pdf", bbox_inches="tight")
        plt.close(figure)

    def sampling_sensitivity(self, bundle: KernelBundle, y_train, y_test):
        rows = []
        rng = np.random.default_rng(self.random_seed)
        for shots in (100, 500, 1000, 5000):
            sampled_train = rng.binomial(shots, np.clip(bundle.train, 0, 1)) / shots
            sampled_test = rng.binomial(shots, np.clip(bundle.test, 0, 1)) / shots
            sampled_train = (sampled_train + sampled_train.T) / 2
            np.fill_diagonal(sampled_train, 1)
            sampled = KernelBundle(sampled_train, sampled_test, bundle.circuit, 0)
            metrics = self.classification_metrics(sampled, y_train, y_test)
            rows.append(
                {
                    "shots": shots,
                    "accuracy": metrics["accuracy"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "f1": metrics["f1"],
                    "kernel_rmse": np.sqrt(np.mean((sampled_train - bundle.train) ** 2)),
                }
            )
        return pd.DataFrame(rows)

    def noise_sensitivity(self, bundle: KernelBundle, y_train, y_test):
        rows = []
        dimension = 2 ** bundle.circuit.num_qubits
        for probability in (0.0, 0.005, 0.01, 0.02, 0.05):
            noisy_train = (1 - probability) * bundle.train + probability / dimension
            noisy_test = (1 - probability) * bundle.test + probability / dimension
            np.fill_diagonal(noisy_train, 1)
            noisy = KernelBundle(noisy_train, noisy_test, bundle.circuit, 0)
            metrics = self.classification_metrics(noisy, y_train, y_test)
            rows.append({"depolarizing_probability": probability, **metrics})
        return pd.DataFrame(rows)

    def save_sensitivity_plot(self, frame: pd.DataFrame, x: str, slug: str, suffix: str):
        melted = frame.melt(
            id_vars=[x],
            value_vars=["accuracy", "balanced_accuracy", "f1"],
            var_name="Métrica",
            value_name="Valor",
        )
        figure, axis = plt.subplots(figsize=(7, 4))
        sns.lineplot(data=melted, x=x, y="Valor", hue="Métrica", marker="o", ax=axis)
        axis.set_ylim(0, 1.02)
        figure.savefig(self.output_dir / f"{slug}_{suffix}.pdf", bbox_inches="tight")
        plt.close(figure)

    def ablations(self, config: dict):
        variants = []
        for reps in (1, 2, 3):
            variants.append(("repeticiones", str(reps), {"reps": reps}))
        for scaling in ("standard_pca", "minmax_0_pi", "minmax_minus_pi_pi"):
            variants.append(("escalado", scaling, {"scaling": scaling}))
        for layers in (1, 2, 3):
            variants.append(("recarga_datos", str(layers), {"reps": layers}))

        if config["featuremap"] != "z_feature_map":
            for topology in ("linear", "circular", "full"):
                variants.append(("entrelazamiento", topology, {"entanglement": topology}))
        if config["featuremap"] == "pauli_feature_map":
            for label, terms in (
                ("Z", ["Z"]),
                ("ZZ", ["ZZ"]),
                ("Z_ZZ", ["Z", "ZZ"]),
                ("X_Y_ZZ", ["X", "Y", "ZZ"]),
            ):
                variants.append(("terminos_pauli", label, {"paulis": terms}))

        rows = []
        seen = set()
        for experiment, value, arguments in variants:
            key = (experiment, value)
            if key in seen:
                continue
            seen.add(key)
            bundle, y_train, y_test = self.generate_bundle(config, **arguments)
            metrics = self.classification_metrics(bundle, y_train, y_test)
            rows.append({"experimento": experiment, "valor": value, **metrics})
        return pd.DataFrame(rows)

    def save_ablation_plot(self, frame: pd.DataFrame, slug: str):
        melted = frame.melt(
            id_vars=["experimento", "valor"],
            value_vars=["accuracy", "balanced_accuracy", "f1"],
            var_name="Métrica",
            value_name="Resultado",
        )
        melted["variante"] = melted["experimento"] + ": " + melted["valor"]
        figure_height = max(5, 0.32 * melted["variante"].nunique())
        figure, axis = plt.subplots(figsize=(8, figure_height))
        sns.barplot(data=melted, y="variante", x="Resultado", hue="Métrica", ax=axis)
        axis.set_xlim(0, 1)
        axis.set_ylabel("")
        figure.savefig(self.output_dir / f"{slug}_ablaciones.pdf", bbox_inches="tight")
        plt.close(figure)

    @staticmethod
    def load_matrix(path: Path):
        frame = pd.read_csv(path)
        if frame.columns[0].startswith("Unnamed"):
            frame = frame.iloc[:, 1:]
        return frame.to_numpy(dtype=float)

    def h2_comparison(self, config: dict):
        slug = self.slug(config)
        generated_bundle, _, _ = self.generate_bundle(config)
        pairs = (
            (
                "entrenamiento",
                self.repo_root / "data" / "kernel" / f"{slug}.csv",
                self.repo_root / "data" / "kernel_h2" / f"{slug}.csv",
                generated_bundle.train,
            ),
            (
                "prueba",
                self.repo_root / "data" / "test" / f"{slug}.csv",
                self.repo_root / "data" / "test_h2" / f"{slug}.csv",
                generated_bundle.test,
            ),
        )
        missing = []

        for group, ideal_path, h2_path, generated_ideal in pairs:
            ideal = self.load_matrix(ideal_path) if ideal_path.exists() else generated_ideal
            figure, axis = plt.subplots(figsize=(6, 5))
            sns.heatmap(ideal, cmap="viridis", vmin=0, vmax=1, square=True, ax=axis)
            axis.set_xlabel("Muestra")
            axis.set_ylabel("Muestra")
            figure.savefig(
                self.output_dir / f"{slug}_{group}_kernel.pdf", bbox_inches="tight"
            )
            plt.close(figure)

            if not h2_path.exists():
                missing.append(str(h2_path))
                continue
            h2 = self.load_matrix(h2_path)
            if ideal.shape != h2.shape:
                raise ValueError(
                    f"Formas incompatibles para {slug} ({group}): "
                    f"{ideal.shape} y {h2.shape}"
                )
            difference = h2 - ideal
            squared = difference**2
            absolute = np.sqrt(squared)
            difference_limit = max(float(np.abs(difference).max()), 1e-12)

            plots = (
                (h2, f"{group}_kernel_h2", "viridis", 0, 1),
                (
                    difference,
                    f"{group}_diferencia_h2_menos_kernel",
                    "vlag",
                    -difference_limit,
                    difference_limit,
                ),
                (squared, f"{group}_error_cuadrado", "magma", 0, squared.max()),
                (
                    absolute,
                    f"{group}_error_absoluto_raiz_error_cuadrado",
                    "magma",
                    0,
                    absolute.max(),
                ),
            )
            for matrix, suffix, cmap, lower, upper in plots:
                figure, axis = plt.subplots(figsize=(6, 5))
                sns.heatmap(
                    matrix, cmap=cmap, vmin=lower, vmax=upper, square=True, ax=axis
                )
                axis.set_xlabel("Muestra")
                axis.set_ylabel("Muestra")
                figure.savefig(
                    self.output_dir / f"{slug}_{suffix}.pdf", bbox_inches="tight"
                )
                plt.close(figure)
        return missing

    def run(self, combinations: list[dict], h2_config: dict):
        summary_rows = []
        for config in combinations:
            slug = self.slug(config)
            print(f"Procesando {slug}")
            bundle, y_train, y_test = self.generate_bundle(config)
            metrics = self.classification_metrics(bundle, y_train, y_test)
            cv_mean, cv_std = self.cross_validation(bundle.train, y_train)
            alignment = self.target_alignment(bundle.train, y_train)
            intra, inter = self.similarities(bundle.train, y_train)
            eigenvalues, effective_rank, numerical_rank = self.spectrum(bundle.train)
            compiled, depth, two_qubit, gate_count = self.circuit_cost(bundle.circuit)

            metrics_row = {
                "N": config["N"],
                "dim": config["dim"],
                "featuremap": config["featuremap"],
                **metrics,
                "cv_balanced_accuracy_mean": cv_mean,
                "cv_balanced_accuracy_std": cv_std,
                "kernel_target_alignment": alignment,
                "intra_class_similarity_mean": float(np.mean(intra)),
                "inter_class_similarity_mean": float(np.mean(inter)),
                "effective_rank": effective_rank,
                "numerical_rank": numerical_rank,
                "circuit_depth": depth,
                "two_qubit_gates": two_qubit,
                "gate_count": gate_count,
                "kernel_generation_seconds": bundle.generation_seconds,
                "train_kernel_circuits": config["N"] * (config["N"] - 1) // 2,
                "test_kernel_circuits": len(y_test) * config["N"],
            }
            summary_rows.append(metrics_row)
            self.save_table(pd.DataFrame([metrics_row]), f"{slug}_metricas_costos.pdf")
            self.save_base_plots(slug, bundle, y_train, eigenvalues)
            self.save_circuits(slug, bundle.circuit, compiled)

            sampling = self.sampling_sensitivity(bundle, y_train, y_test)
            self.save_table(sampling, f"{slug}_sensibilidad_muestreo_datos.pdf")
            self.save_sensitivity_plot(sampling, "shots", slug, "sensibilidad_muestreo")

            noise = self.noise_sensitivity(bundle, y_train, y_test)
            self.save_table(noise, f"{slug}_sensibilidad_ruido_datos.pdf")
            self.save_sensitivity_plot(
                noise, "depolarizing_probability", slug, "sensibilidad_ruido"
            )

            ablation = self.ablations(config)
            self.save_table(ablation, f"{slug}_ablaciones_datos.pdf")
            self.save_ablation_plot(ablation, slug)

        summary = pd.DataFrame(summary_rows)
        self.save_table(summary, "resumen_todas_combinaciones.pdf")
        summary.to_csv(self.output_dir / "resumen_todas_combinaciones.csv", index=False)
        missing_h2 = self.h2_comparison(h2_config)
        return summary, missing_h2

    @staticmethod
    def _short_label(row):
        if row["featuremap"] == "SVM_RBF_sklearn":
            return "SVM-RBF sklearn"
        return f"n{row['N']}_d{row['dim']}_{row['featuremap']}"

    def save_group_metrics(
        self, frame: pd.DataFrame, sklearn_reference: pd.DataFrame, filename: str
    ):
        metric_columns = [
            "expressivity",
            "accuracy",
            "balanced_accuracy",
            "f1",
            "cv_balanced_accuracy",
        ]
        plot_frame = pd.concat([frame.copy(), sklearn_reference.copy()], ignore_index=True)
        plot_frame["combinación"] = plot_frame.apply(self._short_label, axis=1)
        long = plot_frame.melt(
            id_vars="combinación",
            value_vars=metric_columns,
            var_name="Métrica",
            value_name="Valor",
        )
        figure, axis = plt.subplots(figsize=(10, 5))
        sns.barplot(data=long, x="combinación", y="Valor", hue="Métrica", ax=axis)
        axis.set_ylim(0, 1)
        axis.set_xlabel("")
        axis.set_ylabel("Valor")
        axis.tick_params(axis="x", rotation=22)
        axis.legend(loc="upper right")
        figure.savefig(self.output_dir / filename, bbox_inches="tight")
        print(f"\nImagen: {Path(filename).stem}")
        display(figure)
        plt.close(figure)

    def sklearn_rbf_reference(self, seeds=(42, 123, 2026)):
        data = pd.read_csv(self.data_path)
        features = [column for column in data.columns if column != "Potability"]
        x = data[features]
        y = data["Potability"].astype(int)
        rows = []

        for seed in seeds:
            x_train, x_test, y_train, y_test = train_test_split(
                x,
                y,
                test_size=0.2,
                stratify=y,
                random_state=seed,
            )
            pipeline = ImbPipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("smote", SMOTE(random_state=seed)),
                    ("classifier", SVC(kernel="rbf", random_state=seed)),
                ]
            )
            cross_validation = StratifiedKFold(
                n_splits=5, shuffle=True, random_state=seed
            )
            search = GridSearchCV(
                estimator=pipeline,
                param_grid={
                    "classifier__C": [0.1, 1, 10],
                    "classifier__gamma": ["scale", "auto", 0.01],
                },
                scoring={
                    "accuracy": "accuracy",
                    "balanced_accuracy": "balanced_accuracy",
                    "f1": "f1",
                },
                refit="f1",
                cv=cross_validation,
                n_jobs=-1,
            )
            search.fit(x_train, y_train)
            predictions = search.best_estimator_.predict(x_test)
            best_index = search.best_index_
            rows.append(
                {
                    "seed": seed,
                    "accuracy": accuracy_score(y_test, predictions),
                    "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
                    "f1": f1_score(y_test, predictions, zero_division=0),
                    "cv_balanced_accuracy": search.cv_results_[
                        "mean_test_balanced_accuracy"
                    ][best_index],
                    "best_C": search.best_params_["classifier__C"],
                    "best_gamma": search.best_params_["classifier__gamma"],
                }
            )

        seed_results = pd.DataFrame(rows)
        reference = {
            "N": len(data),
            "dim": len(features),
            "featuremap": "SVM_RBF_sklearn",
            "expressivity": np.nan,
        }
        for metric in (
            "accuracy",
            "balanced_accuracy",
            "f1",
            "cv_balanced_accuracy",
        ):
            reference[metric] = float(seed_results[metric].mean())
            reference[f"{metric}_std"] = float(seed_results[metric].std(ddof=1))
        reference["circuit_depth"] = np.nan
        reference["two_qubit_gates"] = np.nan
        return pd.DataFrame([reference]), seed_results

    def save_group_heatmaps(self, rows: pd.DataFrame, bundles: dict, filename: str):
        count = len(rows)
        figure, axes = plt.subplots(1, count, figsize=(5.2 * count, 4.8), squeeze=False)
        image = None
        for axis, (_, row) in zip(axes[0], rows.iterrows()):
            slug = self.slug(row.to_dict())
            image = sns.heatmap(
                bundles[slug].train,
                cmap="viridis",
                vmin=0,
                vmax=1,
                square=True,
                cbar=False,
                ax=axis,
            )
            axis.set_xlabel(self._short_label(row))
            axis.set_ylabel("Muestra")
        if image is not None:
            scalar = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, 1))
            scalar.set_array([])
            figure.colorbar(scalar, ax=list(axes[0]), fraction=0.025, pad=0.02)
        figure.savefig(self.output_dir / filename, bbox_inches="tight")
        print(f"\nImagen: {Path(filename).stem}")
        display(figure)
        plt.close(figure)

    def save_expressive_circuit_table(self, frame: pd.DataFrame):
        table = frame[
            [
                "N",
                "dim",
                "featuremap",
                "expressivity",
                "circuit_depth",
                "two_qubit_gates",
            ]
        ].rename(
            columns={
                "expressivity": "expresividad",
                "circuit_depth": "profundidad",
                "two_qubit_gates": "compuertas_2q",
            }
        )
        self.save_table(
            table,
            "grupo_mayor_expresividad_profundidad_compuertas_2q.pdf",
            show=True,
        )

    def save_h2_absolute_noise_heatmaps(self, config: dict):
        slug = self.slug(config)
        ideal_bundle, _, _ = self.generate_bundle(config)
        pairs = (
            (
                "entrenamiento",
                ideal_bundle.train,
                self.repo_root / "data" / "kernel_h2" / f"{slug}.csv",
            ),
            (
                "prueba",
                ideal_bundle.test,
                self.repo_root / "data" / "test_h2" / f"{slug}.csv",
            ),
        )
        missing = []
        for group, ideal, h2_path in pairs:
            if not h2_path.exists():
                missing.append(str(h2_path))
                continue
            h2 = self.load_matrix(h2_path)
            if ideal.shape != h2.shape:
                raise ValueError(
                    f"Formas incompatibles para {slug} ({group}): "
                    f"{ideal.shape} y {h2.shape}"
                )
            absolute_difference = np.abs(h2 - ideal)

            figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))
            matrices = (ideal, h2, absolute_difference)
            labels = (
                f"{group} kernel ideal",
                f"{group} kernel H2",
                f"{group} |H2 - ideal|",
            )
            for axis, matrix, label in zip(axes, matrices, labels):
                sns.heatmap(
                    matrix,
                    cmap="viridis",
                    vmin=0,
                    vmax=1,
                    square=True,
                    cbar=False,
                    ax=axis,
                )
                axis.set_xlabel(label)
                axis.set_ylabel("Muestra")
            scalar = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, 1))
            scalar.set_array([])
            figure.colorbar(scalar, ax=list(axes), fraction=0.025, pad=0.02)
            figure.savefig(
                self.output_dir / f"{slug}_{group}_kernel_h2_ruido_absoluto.pdf",
                bbox_inches="tight",
            )
            print(f"\nImagen: {slug}_{group}_kernel_h2_ruido_absoluto")
            display(figure)
            plt.close(figure)
        return missing

    def run_focused(self, combinations: list[dict], h2_config: dict, top_k: int = 3):
        rows = []
        bundles = {}
        circuits = {}

        for config in combinations:
            slug = self.slug(config)
            print(f"Procesando {slug}")
            bundle, y_train, y_test = self.generate_bundle(config)
            metrics = self.classification_metrics(bundle, y_train, y_test)
            cv_mean, _ = self.cross_validation(bundle.train, y_train)
            compiled, depth, two_qubit, _ = self.circuit_cost(bundle.circuit)
            rows.append(
                {
                    **config,
                    "expressivity": float(np.mean(bundle.train)),
                    "accuracy": metrics["accuracy"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "f1": metrics["f1"],
                    "cv_balanced_accuracy": cv_mean,
                    "circuit_depth": depth,
                    "two_qubit_gates": two_qubit,
                }
            )
            bundles[slug] = bundle
            circuits[slug] = (bundle.circuit, compiled)

        results = pd.DataFrame(rows)
        top_k = min(top_k, len(results))
        expressive = results.nlargest(top_k, "expressivity").reset_index(drop=True)
        accurate = results.nlargest(top_k, "accuracy").reset_index(drop=True)
        sklearn_reference, sklearn_by_seed = self.sklearn_rbf_reference()

        self.save_group_metrics(
            expressive,
            sklearn_reference,
            "grupo_mayor_expresividad_metricas_vs_sklearn.pdf",
        )
        self.save_group_metrics(
            accurate,
            sklearn_reference,
            "grupo_mayor_accuracy_metricas_vs_sklearn.pdf",
        )
        self.save_group_heatmaps(
            expressive, bundles, "grupo_mayor_expresividad_heatmaps_escala_0_1.pdf"
        )
        self.save_group_heatmaps(
            accurate, bundles, "grupo_mayor_accuracy_heatmaps_escala_0_1.pdf"
        )
        self.save_expressive_circuit_table(expressive)

        for _, row in expressive.iterrows():
            slug = self.slug(row.to_dict())
            original, compiled = circuits[slug]
            self.save_circuits(slug, original, compiled, show=True)

        unified = pd.concat(
            [
                results.assign(modelo="QSVM"),
                sklearn_reference.assign(modelo="SVM clásica"),
            ],
            ignore_index=True,
        )
        unified.to_csv(self.output_dir / "resultados_qsvm_vs_sklearn.csv", index=False)
        sklearn_by_seed.to_csv(
            self.output_dir / "resultados_sklearn_por_semilla.csv", index=False
        )
        missing_h2 = self.save_h2_absolute_noise_heatmaps(h2_config)
        return results, expressive, accurate, sklearn_reference, missing_h2
