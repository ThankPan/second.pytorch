for d in $(ls samples)
do
    for file in $(comm -1 -2 <(ls samples/$d | sort) <(ls test/samples/$d | sort))
    do
        rm -rf samples/$d/$file
    done
done
for d in $(ls sweeps)
do
    for file in $(comm -1 -2 <(ls sweeps/$d | sort) <(ls test/sweeps/$d | sort))
    do
        rm -rf sweeps/$d/$file
    done
done