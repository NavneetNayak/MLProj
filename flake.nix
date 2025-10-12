{
  description = "Python env setup flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs = { self, nixpkgs }: let 
    supportedSystems = [ "aarch64-linux" "aarch64-darwin" "x86_64-linux" "x86_64-darwin" ];
    forAllSystems = function : nixpkgs.lib.genAttrs supportedSystems ( system : function (import nixpkgs { inherit system; } ) );
  in {
    devShells = forAllSystems ( pkgs : {
      default = pkgs.mkShell {
        packages = with pkgs; [
          (python3.withPackages ( ps: with ps;[
            ./requirements.txt
          ] ) )
        ];
      };
    } );
  };
}

