# QStock — Quantum-Optimized Stokvel Management

Stokvels are informal community savings groups used by millions of South Africans to pool money for goals like emergency funds, group investments, or festive-season payouts. But most stokvels are still run manually, contributions tracked in notebooks or WhatsApp groups, payout order decided by trust or memory, and no real visibility into who's falling behind. This creates disputes, mismanagement, and risk of funds being lost or misused, especially as groups grow.

Built for the ADAPT IT Social Good Hackathon (Quantum Computing Track), supported by IBM Research, Wits, and SA QuTI.

## The solution

QStock tracks contributions digitally, holds pooled funds through a partner bank account, and uses two quantum-powered components to make the hard parts fair and reliable at scale:

1. **QAOA payout scheduler** (`quantum/qaoa_scheduler.py`) — models fair payout rotation as a combinatorial optimization problem.
2. **Quantum risk classifier** (`quantum/qml_risk_classifier.py`) — flags members likely to miss payments using a quantum kernel SVM.
3. **Classical app layer** (`classical/app.py`) — a Streamlit demo UI tying it together.

## Technical problem statement

The payout scheduler assigns *N* members to payout positions to minimize a cost function
combining unfairness (gap between a member's urgency/contribution rank and their actual
payout position), risk exposure (weighting against low-reliability members receiving early
payouts), and constraint violations (e.g. no repeat payouts within a cycle). This is a QUBO
whose search space grows exponentially with *N*, solved here with QAOA and benchmarked
against a classical greedy/rank-based heuristic. Member risk flagging is framed separately
as binary classification over behavioral features (consistency, lateness, tenure), solved
with a quantum kernel method (QSVM) and benchmarked against a classical SVM.

## Project structure

```
qstock/
├── requirements.txt
├── quantum/
│   ├── qaoa_scheduler.py       # payout rotation optimization (QAOA)
│   └── qml_risk_classifier.py  # member risk classification (QSVM)
├── classical/
│   └── app.py                  # Streamlit demo app
├── data/                       # sample/synthetic member data
└── notebooks/                  # scratch space for experiments
```

## Setup

1. Create and activate a virtual environment:
   ```
   python -m venv qstock_env
   source qstock_env/bin/activate      # macOS/Linux
   .\qstock_env\Scripts\Activate.ps1   # Windows PowerShell
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the QAOA scheduler demo:
   ```
   python quantum/qaoa_scheduler.py
   ```
4. Run the risk classifier demo:
   ```
   python quantum/qml_risk_classifier.py
   ```
5. Launch the app:
   ```
   streamlit run classical/app.py
   ```

## Team

Add your team's names and roles here for the submission.

## Next steps

- Swap synthetic data in `data/` for real/realistic stokvel data.
- Run on real IBM Quantum hardware via `qiskit-ibm-runtime` (currently set up for local Aer simulation).
- Expand the risk classifier's feature set.
- Wire the classical app up to a real (sandbox) banking-as-a-service API.
