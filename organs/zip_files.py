import os
import zipfile

cif_number = 1

def zip(src, dst):
    zf = zipfile.ZipFile("%s.zip" % (dst), "w", zipfile.ZIP_DEFLATED)
    abs_src = os.path.abspath(src)
    for dirname, subdirs, files in os.walk(src):
        for filename in files:
            absname = os.path.abspath(os.path.join(dirname, filename))
            arcname = absname[len(abs_src) + 1:]
            print ('zipping %s as %s' % (os.path.join(dirname, filename),arcname))
            zf.write(absname, arcname)
    zf.close()


if __name__ == "__main__":
    zip("/home/yuyudhan/abs/ckyc-worker/tmp/"+str(cif_number), '/home/yuyudhan/abs/ckyc-worker/tmp/'+str(cif_number))
