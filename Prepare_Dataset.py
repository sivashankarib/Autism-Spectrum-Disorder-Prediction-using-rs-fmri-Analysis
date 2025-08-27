# Python scientific package
from nilearn.datasets import fetch_abide_pcp


# Download the Dataset
an = 0
if an == 1:
    # We specify the site and number of subjects we want to download
    abide = fetch_abide_pcp(derivatives=['func_preproc'],
                            SITE_ID=['NYU'],
                            n_subjects=3)

    # We look at the available data in this dataset
    print(abide.keys())
    print(abide.description)

    # To get the functional dataset, we have to retrieve the variable 'func_preproc'
    func = abide.func_preproc

    # We can also look at where the data is loaded
    print(func[1])



