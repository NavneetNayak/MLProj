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
      default = let 
        requirements = builtins.filter (x: x != "" && builtins.substring 0 1 x != "#") ( pkgs.lib.strings.splitString "\n" ( builtins.readFile ./requirements.txt ) );
      in pkgs.mkShell {
        packages = with pkgs; [
          (python313.withPackages ( ps : map ( requirement : ps.${requirement}) requirements ) )
        ];
      };
    } );
  };
}

