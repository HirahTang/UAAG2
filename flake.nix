{
  description = "Docker build environment with Colima and GCloud";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            colima
            docker
            docker-buildx
            google-cloud-sdk
            git
          ];
        };

        shellHook = ''
          echo "🐳 Docker Environment Loaded"
          echo "   - Run 'colima start' to boot the VM"
          echo "   - Run 'docker ps' to verify connection"
        '';
      }
    );
}
