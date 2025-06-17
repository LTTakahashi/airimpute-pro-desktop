{ pkgs ? import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/nixos-22.05.tar.gz") {} }:

pkgs.mkShell {
  buildInputs = [
    # Core development tools
    pkgs.python312
    pkgs.python312Packages.pip
    pkgs.python312Packages.numpy
    pkgs.python312Packages.pandas
    pkgs.rustc
    pkgs.cargo
    pkgs.nodejs-18_x
    
    # WebKit GTK 4.0 (uses libsoup2)
    pkgs.webkitgtk_4_0
    pkgs.gtk3
    pkgs.cairo
    pkgs.gdk-pixbuf
    pkgs.glib
    pkgs.libsoup
    
    # Build dependencies
    pkgs.pkg-config
    pkgs.openssl
    pkgs.sqlite
    pkgs.gcc
    pkgs.gnumake
    
    # Additional Tauri dependencies
    pkgs.atk
    pkgs.pango
    pkgs.librsvg
    pkgs.libayatana-appindicator
    
    # X11/Wayland support
    pkgs.xorg.libX11
    pkgs.wayland
  ];

  shellHook = ''
    echo "Entering Nix development shell for AirImpute Pro..."
    echo "This environment provides webkit2gtk-4.0 with libsoup2 compatibility"
    echo ""
    export PYTHONDONTWRITEBYTECODE=1
    export PYTHONUNBUFFERED=1
    export PYO3_PYTHON="${pkgs.python312}/bin/python3.12"
    export RUST_BACKTRACE=1
    
    # Ensure npm uses local node_modules
    export PATH="$PWD/node_modules/.bin:$PATH"
    
    echo "Run 'npm run tauri dev' to start the application"
  '';
}