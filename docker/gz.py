import gzip, shutil
with open('harmonics.tar', 'rb') as f_in, gzip.open('harmonics.tar.gz', 'wb') as f_out: 
    shutil.copyfileobj(f_in, f_out)

# close the file
f_in.close()
f_out.close()