# This and add jupyter notebook:

let
  pkgs = import <nixpkgs> {};
in pkgs.mkShell {
  packages = [
    #qt
    pkgs.xcbuild
    pkgs.xcbutilxrm
    pkgs.qt5.qtbase
    pkgs.qt5.qttools
    pkgs.qt5.qtdeclarative
    pkgs.qt5.qtquickcontrols
    pkgs.qt5.qtquickcontrols2
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
    ]))
  ];
}

