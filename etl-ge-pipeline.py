# etl_ge_dual_pipeline.py
# ---------------------------------------------------------
# ETL + Great Expectations para dos datasets:
#  - customer_data.csv (clientes)
#  - retail_data.csv   (ventas/transacciones)
#
# Python 3.10+, GE 0.17+
# ---------------------------------------------------------
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

import great_expectations as ge
from great_expectations.data_context import DataContext
from great_expectations.core.batch import RuntimeBatchRequest

# =============== Regex y helpers de montos ===================
_DECIMAL_RX = re.compile(r"^\s*-?\d{1,3}(\.\d{3})*(,\d+)?\s*$|^\s*-?\d+(\.\d+)?\s*$")

def coerce_amount_series(s: pd.Series) -> pd.Series:
    """
    Convierte strings con formato LATAM/EU (1.234,56) o US (1,234.56) a float.
    - Elimina separadores de miles.
    - Usa punto como separador decimal final.
    - Valores inválidos -> NaN.
    """
    s2 = s.astype(str).str.strip()
    mask_like_number = s2.str.match(_DECIMAL_RX, na=False)

    # US: 1,234.56 -> quitar comas miles
    s2_us = s2.str.replace(",", "", regex=False)
    out = pd.to_numeric(s2_us.where(mask_like_number, s2), errors="coerce")

    # LATAM/EU: 1.234,56 -> quitar puntos miles y coma -> punto
    mask_nan = out.isna()
    if mask_nan.any():
        s2_latam = s2.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
        out.loc[mask_nan] = pd.to_numeric(s2_latam[mask_nan], errors="coerce")

    return out

# ========================= Utilitarios =========================
def print_header(title: str):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

def ensure_datetime_any(series: pd.Series) -> pd.Series:
    """Convierte a datetime con inferencia de formatos y coerción a NaT si falla."""
    return pd.to_datetime(series, errors="coerce", infer_datetime_format=True)

def ensure_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

def init_ge_context(context_dir: str = "great_expectations") -> DataContext:
    os.makedirs(context_dir, exist_ok=True)
    context = ge.get_context(context_root_dir=os.path.abspath(context_dir))
    ds_name = "pandas_runtime"
    existing = [ds["name"] for ds in context.list_datasources()]
    if ds_name not in existing:
        context.add_datasource(
            name=ds_name,
            class_name="Datasource",
            execution_engine={"class_name": "PandasExecutionEngine"},
            data_connectors={
                "runtime_connector": {
                    "class_name": "RuntimeDataConnector",
                    "batch_identifiers": ["default_identifier_name"],
                }
            },
        )
    return context

def runtime_batch_request(df: pd.DataFrame, asset_name: str, ds_name: str = "pandas_runtime"):
    return RuntimeBatchRequest(
        datasource_name=ds_name,
        data_connector_name="runtime_connector",
        data_asset_name=asset_name,
        runtime_parameters={"batch_data": df},
        batch_identifiers={"default_identifier_name": f"{asset_name}_run"},
    )

def run_checkpoint(context, checkpoint_name: str, suite_name: str, batch_request):
    # No persistimos validations (tienen batch_data runtime)
    context.add_or_update_checkpoint(
        name=checkpoint_name,
        config_version=1.0,
        class_name="SimpleCheckpoint",
    )
    results = context.run_checkpoint(
        checkpoint_name=checkpoint_name,
        validations=[{
            "batch_request": batch_request,
            "expectation_suite_name": suite_name
        }],
    )
    context.build_data_docs()
    return results

# ========================= EDA =========================
def eda_quick(df: pd.DataFrame, title: str):
    print_header(f"EDA - {title} :: INFO")
    print(df.info())
    print_header(f"EDA - {title} :: DESCRIBE (incl. objetos)")
    print(df.describe(include="all"))
    print_header(f"EDA - {title} :: Nulos por columna")
    print(df.isnull().sum().sort_values(ascending=False))
    print_header(f"EDA - {title} :: Duplicados (filas completas)")
    print(df.duplicated().sum())

# ========================= Fix columnas retail =========================
def fix_retail_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Corrige el error del dataset informado por el profesor:
      - 'id'            -> 'customer_id'       (ID del cliente)
      - 'customer_id'   -> 'transaction_id'    (ID de la compra)
    """
    d = df.copy()
    cols = {c.strip() for c in d.columns}
    if "id" in cols and "customer_id" in cols:
        d = d.rename(columns={"id": "customer_id", "customer_id": "transaction_id"})
    return d

# ========================= Limpieza =========================
def clean_customers(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d = d.drop_duplicates()
    if "id" in d.columns:
        d = d.drop_duplicates(subset=["id"], keep="first")

    if "signup_date" in d.columns:
        d["signup_date"] = ensure_datetime_any(d["signup_date"])

    if "age" in d.columns:
        d["age"] = ensure_numeric(d["age"])
        d.loc[(d["age"] < 0) | (d["age"] > 120), "age"] = np.nan

    if "gender" in d.columns:
        d["gender"] = d["gender"].astype(str).str.strip().str.title()
        map_gender = {"F": "Female", "M": "Male", "Fem": "Female", "Masc": "Male",
                      "Female": "Female", "Male": "Male", "Nan": "Unknown", "None": "Unknown"}
        d["gender"] = d["gender"].replace(map_gender)
        d.loc[~d["gender"].isin(["Male", "Female", "Other", "Unknown"]), "gender"] = "Unknown"

    if "id" in d.columns:
        d = d[d["id"].notna()].copy()
    return d

def clean_retail(df: pd.DataFrame, valid_customer_ids=None, impute_amount: bool = True) -> pd.DataFrame:
    """
    Limpieza reforzada para retail (pensada para pasar retail_output_suite):
    - transaction_id: str, strip, no nulo, único.
    - purchase_date: datetime, descarta NaT.
    - amount: coerce robusto, negativos->NaN; imputación por mediana (opcional) o drop; sin inf; float y >= 0.
    - customer_id: coerce a Int64 si es posible; descarta nulos; filtra por FK válida si se provee.
    - product_category: strip + title.
    """
    d = df.copy()
    d.columns = [c.strip() for c in d.columns]

    # ---- transaction_id como PK ----
    if "transaction_id" in d.columns:
        d["transaction_id"] = d["transaction_id"].astype(str).str.strip()
    d = d.drop_duplicates()
    if "transaction_id" in d.columns:
        d = d.drop_duplicates(subset=["transaction_id"], keep="first")

    # ---- fecha ----
    if "purchase_date" in d.columns:
        d["purchase_date"] = ensure_datetime_any(d["purchase_date"])
        d = d[d["purchase_date"].notna()].copy()

    # ---- monto ----
    if "amount" in d.columns:
        d["amount"] = coerce_amount_series(d["amount"])
        d.loc[d["amount"] < 0, "amount"] = np.nan

        if impute_amount:
            med = d["amount"].median(skipna=True)
            if pd.isna(med) or med < 0:
                med = 0.0
            d["amount"] = d["amount"].fillna(med)
        else:
            d = d[d["amount"].notna()].copy()

        d = d.replace([np.inf, -np.inf], np.nan)
        d = d[d["amount"].notna()].copy()
        d["amount"] = d["amount"].astype(float)
        d = d[d["amount"] >= 0].copy()

    # ---- FK cliente ----
    if "customer_id" in d.columns:
        try:
            d["customer_id"] = pd.to_numeric(d["customer_id"], errors="coerce").astype("Int64")
        except Exception:
            d["customer_id"] = d["customer_id"].astype(str).str.strip()
        d = d[d["customer_id"].notna()].copy()

        if valid_customer_ids is not None and len(valid_customer_ids) > 0:
            try:
                valid_ids = set(pd.Series(list(valid_customer_ids)).astype("Int64").dropna().tolist())
                d = d[d["customer_id"].astype("Int64").isin(valid_ids)]
            except Exception:
                valid_ids = set(map(str, list(valid_customer_ids)))
                d = d[d["customer_id"].astype(str).isin(valid_ids)]

    # ---- categoría ----
    if "product_category" in d.columns:
        d["product_category"] = d["product_category"].astype(str).str.strip().str.title()

    # Re-chequeo duplicados por transaction_id
    if "transaction_id" in d.columns:
        d = d.drop_duplicates(subset=["transaction_id"], keep="first")

    return d.reset_index(drop=True)

# ========================= Expectation Suites =========================
def policies_customers(validator):
    cols = set(validator.active_batch.head(0).columns.tolist())

    if "id" in cols:
        validator.expect_column_values_to_not_be_null("id")
        validator.expect_column_values_to_be_unique("id")

    if "signup_date" in cols:
        validator.expect_column_values_to_not_be_null("signup_date", mostly=0.95)

    if "email" in cols:
        validator.expect_column_values_to_match_regex("email", r"^[^@\s]+@[^@\s]+\.[^@\s]+$", mostly=0.9)

    if "age" in cols:
        validator.expect_column_values_to_be_between("age", min_value=0, max_value=120, mostly=0.99)

    if "gender" in cols:
        validator.expect_column_values_to_be_in_set("gender", ["Male", "Female", "Other", "Unknown"], mostly=0.99)

    validator.save_expectation_suite(discard_failed_expectations=False)

def policies_retail_input(validator):
    """Reglas suaves para datos crudos (que NO tumben el run)."""
    cols = set(validator.active_batch.head(0).columns.tolist())

    if "transaction_id" in cols:
        validator.expect_column_values_to_not_be_null("transaction_id", mostly=0.90)
        validator.expect_column_values_to_be_unique("transaction_id", mostly=0.90)

    if "purchase_date" in cols:
        validator.expect_column_values_to_not_be_null("purchase_date", mostly=0.70)

    if "amount" in cols:
        # Muy suave: que "parezca" numérico al menos en 60% del dataset
        validator.expect_column_values_to_be_in_type_list(
            "amount", ["int", "int64", "float", "float64", "double"], mostly=0.60
        )

    if "customer_id" in cols:
        validator.expect_column_values_to_not_be_null("customer_id", mostly=0.90)

    validator.save_expectation_suite(discard_failed_expectations=False)



def policies_retail_output(validator, customers_clean=None):
    """Reglas estrictas para datos limpios."""
    cols = set(validator.active_batch.head(0).columns.tolist())

    if "transaction_id" in cols:
        validator.expect_column_values_to_not_be_null("transaction_id")
        validator.expect_column_values_to_be_unique("transaction_id")

    if "purchase_date" in cols:
        validator.expect_column_values_to_not_be_null("purchase_date")

    if "amount" in cols:
        validator.expect_column_values_to_not_be_null("amount")
        validator.expect_column_values_to_be_between("amount", min_value=0)

    if "customer_id" in cols:
        validator.expect_column_values_to_not_be_null("customer_id")
        if customers_clean is not None:
            valid_ids_str = set(customers_clean["id"].astype(str).str.strip())
            validator.expect_column_values_to_be_in_set(
                "customer_id", list(valid_ids_str), mostly=0.995
            )

    validator.save_expectation_suite(discard_failed_expectations=False)

# ========================= Visualizaciones & KPIs =========================
def visuals_and_kpis(customers: pd.DataFrame, retail: pd.DataFrame, out_dir="outputs"):
    os.makedirs(out_dir, exist_ok=True)

    # ==== Preparación y helpers ====
    r = retail.copy()
    c = customers.copy()

    # Campos clave para "validez financiera"
    required_cols = [col for col in ["transaction_id", "purchase_date", "amount", "customer_id"] if col in r.columns]
    n_total = len(r)

    # ------------------------------
    # a) Ensure Financial Integrity
    # ------------------------------
    kpi_fin_integrity = None
    if n_total > 0 and set(["purchase_date", "amount", "customer_id"]).issubset(r.columns):
        # inválida si hay NaN en requeridas o amount < 0
        invalid_mask = r[required_cols].isna().any(axis=1)
        if "amount" in r.columns:
            invalid_mask |= (r["amount"] < 0)
        n_invalid = int(invalid_mask.sum())
        n_valid = int(n_total - n_invalid)
        kpi_fin_integrity = (n_valid / n_total) * 100

        plt.figure()
        plt.bar(["Válidas", "Inválidas"], [n_valid, n_invalid])
        plt.title("Integridad financiera: transacciones válidas vs inválidas")
        plt.ylabel("Cantidad de transacciones")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "a_integridad_financiera.png"))
        plt.close()

    # -----------------------------------------
    # b) Support Strategic Planning (ventas mes)
    # -----------------------------------------
    kpi_sales_total = None
    if {"purchase_date", "amount"}.issubset(r.columns) and len(r) > 0:
        r["month"] = r["purchase_date"].dt.to_period("M")
        monthly = r.groupby("month")["amount"].sum().sort_index()
        kpi_sales_total = float(monthly.sum())

        plt.figure()
        monthly.plot(kind="bar", title="Ventas mensuales (monto total)")
        plt.ylabel("Monto")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "b_ventas_mensuales.png"))
        plt.close()

    # ----------------------------------------------------
    # c) Strengthening Customer & Product Insights (Top-5)
    # ----------------------------------------------------
    kpi_top_cat = None
    if "product_category" in r.columns and "amount" in r.columns:
        top_cat = r.groupby("product_category")["amount"].sum().sort_values(ascending=False).head(5)
        if len(top_cat) > 0:
            kpi_top_cat = top_cat.index.tolist()

            plt.figure()
            top_cat.sort_values().plot(kind="barh", title="Top 5 categorías por ventas (monto)")
            plt.xlabel("Monto")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "c_top_categorias.png"))
            plt.close()

    # (extra insight cliente) Ventas por género
    if {"customer_id", "amount"}.issubset(r.columns) and {"id", "gender"}.issubset(c.columns):
        merged = r.merge(c[["id", "gender"]].rename(columns={"id": "customer_id"}),
                         on="customer_id", how="left")
        if "gender" in merged.columns:
            plt.figure()
            merged.groupby("gender")["amount"].sum().sort_values(ascending=False).plot(
                kind="bar", title="Ventas por género"
            )
            plt.ylabel("Monto total")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "c_ventas_por_genero.png"))
            plt.close()

    # ------------------------------------------------------
    # d) Transparent & Defensible Reporting (FK compliance)
    # ------------------------------------------------------
    kpi_fk = None
    if {"customer_id"}.issubset(r.columns) and {"id"}.issubset(c.columns):
        valid_ids = set(c["id"].astype(str).str.strip())
        fk_ratio = r["customer_id"].astype(str).str.strip().isin(valid_ids).mean() * 100
        kpi_fk = fk_ratio

        plt.figure()
        plt.pie([fk_ratio, 100 - fk_ratio],
                labels=["Válidas", "No válidas"], autopct="%1.1f%%", startangle=90)
        plt.title("Integridad referencial (FK clientes)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "d_integridad_fk.png"))
        plt.close()

    # (se mantiene tu distribución de género de clientes)
    if "gender" in c.columns:
        g = c["gender"].value_counts(dropna=False).sort_index()
        if len(g) > 0:
            plt.figure()
            g.plot(kind="bar", title="Distribución de género (clientes)")
            plt.ylabel("Cantidad de clientes")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "dist_genero_clientes.png"))
            plt.close()

    # ------------------------------
    # KPIs a archivo de texto
    # ------------------------------
    with open(os.path.join(out_dir, "kpis.txt"), "w", encoding="utf-8") as f:
        f.write("KPIs por objetivo de negocio\n")
        f.write("=====================================\n")

        if kpi_fin_integrity is not None:
            f.write(f"a) Integridad financiera (%% válidas): {kpi_fin_integrity:.2f}%  -> a_integridad_financiera.png\n")
        else:
            f.write("a) Integridad financiera: N/D\n")

        if kpi_sales_total is not None:
            f.write(f"b) Ventas totales del período: {kpi_sales_total:,.2f}  -> b_ventas_mensuales.png\n")
        else:
            f.write("b) Ventas totales del período: N/D\n")

        if kpi_top_cat is not None:
            f.write(f"c) Top 5 categorías por ventas: {', '.join(kpi_top_cat)}  -> c_top_categorias.png\n")
        else:
            f.write("c) Top 5 categorías por ventas: N/D\n")

        if kpi_fk is not None:
            f.write(f"d) Integridad referencial (FK válida): {kpi_fk:.2f}%  -> d_integridad_fk.png\n")
        else:
            f.write("d) Integridad referencial (FK válida): N/D\n")

        # Métrica de completitud general (como tenías)
        completeness = (1 - r.isnull().any(axis=1).mean()) * 100 if len(r) else 0.0
        f.write(f"\nCompletitud retail (filas sin NaN): {completeness:.2f}%\n")

    # ------------------------------
    # Log de archivos generados
    # ------------------------------
    print_header("KPIs y visualizaciones generadas")
    for fn in [
        "a_integridad_financiera.png",
        "b_ventas_mensuales.png",
        "c_top_categorias.png",
        "c_ventas_por_genero.png",
        "d_integridad_fk.png",
        "dist_genero_clientes.png",
        "kpis.txt",
    ]:
        path = os.path.join(out_dir, fn)
        if os.path.exists(path):
            print("->", path)


    

# ========================= MAIN =========================
def main():
    parser = argparse.ArgumentParser(description="ETL + GE (customers & retail)")
    parser.add_argument("--customers", type=str, required=True, help="Ruta a customer_data.csv")
    parser.add_argument("--retail", type=str, required=True, help="Ruta a retail_data.csv")
    parser.add_argument("--context_dir", type=str, default="great_expectations", help="Carpeta GE")
    parser.add_argument("--open_docs", action="store_true", help="Abrir Data Docs")
    args = parser.parse_args()

    # ---------- Carga ----------
    customers_raw = pd.read_csv(args.customers)
    retail_raw = pd.read_csv(args.retail)
    retail_raw = fix_retail_columns(retail_raw)  # <- corrección del profesor

    # ---------- EDA ----------
    eda_quick(customers_raw, "CUSTOMERS (RAW)")
    eda_quick(retail_raw, "RETAIL (RAW)")

    # ---------- GE Context ----------
    context = init_ge_context(args.context_dir)

    # ---------- Suites INPUT ----------
    suite_customers_in = "customers_input_suite"
    suite_retail_in = "retail_input_suite"
    context.add_or_update_expectation_suite(suite_customers_in)
    context.add_or_update_expectation_suite(suite_retail_in)

    br_customers_in = runtime_batch_request(customers_raw, "customers_raw")
    br_retail_in = runtime_batch_request(retail_raw, "retail_raw")

    v_c_in = context.get_validator(batch_request=br_customers_in, expectation_suite_name=suite_customers_in)
    v_r_in = context.get_validator(batch_request=br_retail_in, expectation_suite_name=suite_retail_in)

    policies_customers(v_c_in)
    policies_retail_input(v_r_in)

    print_header("GE INPUT - Running checkpoints")
    res_c_in = run_checkpoint(context, "chk_customers_input", suite_customers_in, br_customers_in)
    res_r_in = run_checkpoint(context, "chk_retail_input", suite_retail_in, br_retail_in)
    print("Customers INPUT:", res_c_in.list_validation_result_identifiers())
    print("Retail    INPUT:", res_r_in.list_validation_result_identifiers())

    # ---------- Limpieza ----------
    customers_clean = clean_customers(customers_raw)
    if "id" in customers_clean.columns:
        customers_clean["id"] = customers_clean["id"].astype(str).str.strip()

    # Set de IDs válidos como string (estable)
    valid_customer_ids_str = set(customers_clean["id"].astype(str).str.strip()) if "id" in customers_clean else set()

    retail_clean = clean_retail(retail_raw, valid_customer_ids=valid_customer_ids_str, impute_amount=True)
    if "customer_id" in retail_clean.columns:
        retail_clean["customer_id"] = retail_clean["customer_id"].astype(str).str.strip()

    # ---------- Suites OUTPUT ----------
    suite_customers_out = "customers_output_suite"
    suite_retail_out = "retail_output_suite"
    context.add_or_update_expectation_suite(suite_customers_out)
    context.add_or_update_expectation_suite(suite_retail_out)
    try:
        context.delete_expectation_suite(suite_retail_out)
    except Exception:
        pass
    context.create_expectation_suite(suite_retail_out, overwrite_existing=True)

    context.add_or_update_expectation_suite(suite_customers_out)

    br_customers_out = runtime_batch_request(customers_clean, "customers_clean")

    print("\n>>> Diagnóstico previo a OUTPUT (tipos y FK)")
    print("retail_clean.customer_id dtype:", retail_clean["customer_id"].dtype if "customer_id" in retail_clean else "no-col")
    print("customers_clean.id dtype:", customers_clean["id"].dtype if "id" in customers_clean else "no-col")
    fk_ok = retail_clean["customer_id"].isin(set(customers_clean["id"])).mean() * 100
    print(f"FK OK (string vs string): {fk_ok:.2f}%")  # debería ser ~100%

    br_retail_out = runtime_batch_request(retail_clean, "retail_clean")

    v_c_out = context.get_validator(batch_request=br_customers_out, expectation_suite_name=suite_customers_out)
    v_r_out = context.get_validator(batch_request=br_retail_out, expectation_suite_name=suite_retail_out)

    policies_customers(v_c_out)
    policies_retail_output(v_r_out, customers_clean=customers_clean)

    print_header("GE OUTPUT - Running checkpoints")
    res_c_out = run_checkpoint(context, "chk_customers_output", suite_customers_out, br_customers_out)
    res_r_out = run_checkpoint(context, "chk_retail_output", suite_retail_out, br_retail_out)
    print("Customers OUTPUT:", res_c_out.list_validation_result_identifiers())
    print("Retail    OUTPUT:", res_r_out.list_validation_result_identifiers())

    # ---------- Data Docs ----------
    context.build_data_docs()
    if args.open_docs:
        context.open_data_docs()

    # ---------- Visualizaciones & KPIs ----------
    os.makedirs("outputs", exist_ok=True)
    visuals_and_kpis(customers_clean, retail_clean, out_dir="outputs")

    print_header("FIN")
    print("Data Docs: great_expectations/uncommitted/data_docs/")
    print("Outputs: carpeta 'outputs/'")

if __name__ == "__main__":
    main()
