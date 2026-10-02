from setuptools import setup
package_name = 'ur_perception'
setup(
    name=package_name, version='0.1.0', packages=[package_name],
    data_files=[('share/ament_index/resource_index/packages', ['resource/' + package_name]),
                ('share/' + package_name, ['package.xml'])],
    install_requires=['setuptools'], zip_safe=True,
    maintainer='you', maintainer_email='you@example.com',
    description='ur_perception', license='MIT',
    entry_points={'console_scripts': [
        'detect_objects = ur_perception.detect_objects:main',
        'detect_open = ur_perception.detect_open:main',
        'detect_open_check = ur_perception.detect_open_check:main']},
)
