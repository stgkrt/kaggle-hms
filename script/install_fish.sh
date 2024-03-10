apt-get update --quiet && \
apt-get install --quiet --yes software-properties-common openssh-client git && \
add-apt-repository --yes ppa:fish-shell/release-3 && \
apt-get install --quiet --yes fish

chsh -s /usr/bin/fish
SHELL /usr/bin/fish
LANG=C.UTF-8 LANGUAGE=C.UTF-8 LC_ALL=C.UTF-8