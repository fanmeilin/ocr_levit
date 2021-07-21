# dvc init
# dvc remote add -d myremote s3://dvc
# dvc remote modify myremote endpointurl http://ceph01

set -x

# update git submodule
git submodule update --init --recursive

# download data
export AWS_ACCESS_KEY_ID=18M4BI7CGWKWSYELOXC3
export AWS_SECRET_ACCESS_KEY=zHpZOndT6y4Dkz5GkH4g1DAkyRuDKe7BUFSjbV3b

cd lib/classifier/ && ./setup.sh && cd ..
cd detector/ && ./setup.sh release && cd ..
cd ..

case $1 in
    ("release" | "distribution") \
			# submodule setup
			echo "in release mode";;
    (*) \
			# submodule setup
            yes | dvc pull -f;;
esac
