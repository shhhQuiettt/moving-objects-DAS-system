# This and add jupyter notebook:

let
  pkgs = import <nixpkgs> {};
in pkgs.mkShell {
  packages = [
    #qt
    #pkgs.jupyter

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
      python-pkgs.pyqt5
      python-pkgs.virtualenv
    ]))
    pkgs.jupyter
    (pkgs.rWrapper.override {
      packages = with pkgs.rPackages; [
        ggplot2
        knitr
        readr
        reticulate
      ];
    })
  ];
}

