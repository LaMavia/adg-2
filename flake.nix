{
  inputs = {
    utils.url = "github:numtide/flake-utils";
  };
  outputs = { self, nixpkgs, utils }: utils.lib.eachDefaultSystem (system:
    let
      pkgs = nixpkgs.legacyPackages.${system};
    in
    {
      devShell = pkgs.mkShell {
        buildInputs = (with pkgs.python312Packages; [
          python
          numpy
          biopython
          mmh3
          pandas
          scikit-learn
          tqdm
        ]);
      };
    }
  );
}
