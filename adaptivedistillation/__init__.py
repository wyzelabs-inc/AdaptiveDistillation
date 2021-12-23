import mmcv
import mmcls


def digit_version(version_str):
    digit_version = []
    for x in version_str.split('.'):
        if x.isdigit():
            digit_version.append(int(x))
        elif x.find('rc') != -1:
            patch_version = x.split('rc')
            digit_version.append(int(patch_version[0]) - 1)
            digit_version.append(int(patch_version[1]))
    return digit_version


mmcv_minimum_version = '1.3.1'
mmcv_maximum_version = '1.5.0'
mmcv_version = digit_version(mmcv.__version__)

mmcls_required_version = '0.13.0'
mmcls_version = digit_version(mmcls.__version__)


assert (mmcv_version >= digit_version(mmcv_minimum_version)
        and mmcv_version <= digit_version(mmcv_maximum_version)), \
    f'MMCV=={mmcv.__version__} is used but incompatible. ' \
    f'Please install mmcv>={mmcv_minimum_version}, <={mmcv_maximum_version}.'

assert mmcls_version == digit_version(mmcls_required_version), \
    f'MMCLS=={mmcls.__version__} is used but incompatible. ' \
        f'Please install mmcls=={mmcls_required_version}'

