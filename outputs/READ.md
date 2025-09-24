# ETL Lab 6 – Pipeline con Great Expectations
- EMILIO MARQUEZ
- SAMUEL URIBE
- JUAN PABLO LOPEZ
Este proyecto implementa un **pipeline ETL** para dos datasets:
- `customer_data.csv` (clientes)
- `retail_data.csv` (ventas/transacciones)

Valida la **calidad de datos** con **Great Expectations (GE)** en dos etapas:
1) **Input (raw)**: reglas *suaves* que describen la “suciedad” del origen.  
2) **Output (clean)**: reglas *estrictas* tras la limpieza (lo que debe quedar “en verde”).

Incluye:
- Corrección automática de nombres de columnas del retail según la indicación del profesor:
  - `id`            → `customer_id` (id de cliente)
  - `customer_id`   → `transaction_id` (id de la transacción)
- Conversión robusta de montos con separadores US/LATAM.
- Integridad referencial: `retail.customer_id ∈ customers.id`.
- KPIs y gráficos exportados a `outputs/`.

---

## Requisitos

- **Python 3.11** (recomendado).  
  > En Windows/macOS/Linux funcionan bien los wheels para las versiones fijadas en `requirements.txt`.

Instala dependencias dentro de un entorno virtual:

```bash
# Windows PowerShell (en la carpeta del proyecto)
py -3.11 -m venv .venv
.\.venv\Scripts\Activate

# Actualiza herramientas base
python -m pip install --upgrade pip setuptools wheel

# Instala dependencias del proyecto
pip install -r requirements.txt

#corre el codigo 
python .\etl-ge-pipeline.py --customers .\data\customer_data.csv --retail .\data\retail_data.csv --open_docs    
