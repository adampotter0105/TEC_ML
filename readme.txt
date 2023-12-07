For each file, enter directory

1) data_fetcher.ipynb : Collects TEC measurement data as JSONs and corresponding structure data as CIF files in /data_linear and /data_volume
2) supercell_generator.ipynb : Takes CIF files and build supercells with deterministic atoms with a dimension scaling with complexity and saves CIFs to /supercells_data
3) label_generator.ipynb : Filters and standardizes TEC measurement data to be linear bulk TEC in [10e-6/K], finds average temperature measured, saves in /labels
4) feature_generator.ipynb : Generates physics-informed features from CIFs in /supercells_data in batches in saves to /features
5) feature_selection.ipynb : Visualizes features in the form of correlation matrices and violin plots saved to /figs
6) train_model.ipynb : Assembles the data from /features and labels from /labels and saves as pytorch tensors in /features. Defines a pytorch model and trains it saving plots to /figs
