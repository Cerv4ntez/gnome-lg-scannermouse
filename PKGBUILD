# Maintainer: Cerv4ntez <18594385+Cerv4ntez@users.noreply.github.com>
pkgname=gnome-lg-scannermouse
pkgver=0.1.1
pkgrel=1
pkgdesc='GNOME app for the LG Smart Scan mouse (LSM-100/LSM-150)'
arch=('any')
url='https://github.com/Cerv4ntez/gnome-lg-scannermouse'
license=('GPL-3.0-or-later')
depends=('python' 'python-gobject' 'python-cairo' 'python-numpy' 'python-scipy' 'python-opencv'
         'gtk4' 'libadwaita' 'gstreamer' 'gst-plugins-base' 'gst-plugins-good')
makedepends=('git' 'python-build' 'python-installer' 'python-setuptools' 'python-wheel')
optdepends=('tesseract: text recognition and upright detection'
            'tesseract-data-osd: turn sideways or upside-down pages upright'
            'tesseract-data-eng: English text recognition'
            'tesseract-data-deu: German text recognition')
source=("git+$url.git#tag=v$pkgver")
sha256sums=('SKIP')

build() {
  cd "$pkgname"
  python -m build --wheel --no-isolation
}

package() {
  cd "$pkgname"
  python -m installer --destdir="$pkgdir" dist/*.whl
  install -d "$pkgdir/usr/share/icons"
  install -Dm644 data/io.github.scannermouse.ScannerMouse.desktop -t "$pkgdir/usr/share/applications"
  cp -r data/icons/hicolor "$pkgdir/usr/share/icons/"
  install -Dm644 README.md -t "$pkgdir/usr/share/doc/$pkgname"
}
