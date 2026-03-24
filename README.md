
# Nonlinear Pushover Dashboard v3

This Streamlit app adds the features you requested:

- user-defined plastic moment hinge properties
- user-defined shear hinge properties
- gravity load per frame/storey
- modal analysis with 1st, 2nd, ... modes
- mode-shape, triangular, or uniform pushover loading
- capacity curve and ADRS capacity spectrum view
- hinge-state visualization
- CSV export

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes

This version is a **professional screening / teaching tool**. It is stronger than a basic demo, but it is still not a full SAP2000-equivalent nonlinear FEM engine.

What is approximated:
- storey stiffness from frame-column sway stiffness proxy
- hinge state from drift/rotation demand proxies
- capacity spectrum method shown as an ADRS / demand proxy intersection
- gravity load affects mass and P-Delta-style stiffness reduction

## Best next upgrades

- true element-by-element nonlinear frame stiffness update
- separate beam/column end hinge assignment by member
- displacement control with Newton-Raphson iteration
- explicit gravity load case before lateral pushover
- FEMA 356 / ASCE 41 backbone input
- panel zone / joint shear logic
- P-M-M interaction for columns
