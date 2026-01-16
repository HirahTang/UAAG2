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
          ];
        };

        shellHook = ''
          if ! colima status > /dev/null 2>&1; then
            colima start
          fi
          echo "🐳 Docker is ready!"
        '';
      }
    );
}
