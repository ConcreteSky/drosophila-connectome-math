# Scientific Data Sources & Schema

## Primary Datasets
1. **`Completeness_783.csv`** (`philshiu/Drosophila_brain_model`)
   - Defines the contiguous index (`0 .. 138,638`) for the `138,639` proofread FlyWire v783 neurons.
2. **`Connectivity_783.parquet`** (`philshiu/Drosophila_brain_model`)
   - Directed synaptic edges (`Presynaptic_Index`, `Postsynaptic_Index`, `Connectivity`, `Excitatory`, `Excitatory x Connectivity`), totaling `15,091,983` directed edges and `54,492,922` synapses.
3. **`Supplemental_file1_neuron_annotations.tsv`** (`flyconnectome/flywire_annotations`)
   - Provides `root_id`, 3D coordinates (`pos_x`, `pos_y`, `pos_z`), `super_class`, `cell_class`, `cell_type`, `top_nt`, and `side`.

## Serverless Subnetwork Export (`artifacts/serverless/`)
- `subnetwork.npz`: Induced 3,000-landmark dorsal subnetwork (`61,453` directed edges) in SciPy CSR format.
- `layout_and_summary.json`: Normalized 2D coordinates (`[0, 1]^2`) across 6 anatomical regions and DuckDB whole-brain summary statistics.
