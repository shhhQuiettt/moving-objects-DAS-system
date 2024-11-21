# This and add jupyter notebook:

let
  pkgs = import <nixpkgs> {};
in pkgs.mkShell {
  packages = [
    (pkgs.python3.withPackages (python-pkgs: [
      python-pkgs.pandas
      python-pkgs.matplotlib
      python-pkgs.opencv4
      python-pkgs.pillow
      python-pkgs.jupyter
      python-pkgs.plotly
      python-pkgs.scikit-image
      python-pkgs.seaborn
      python-pkgs.scikit-learn
    ]))
  ];
}

