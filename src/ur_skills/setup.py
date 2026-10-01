from setuptools import setup
package_name = 'ur_skills'
setup(
    name=package_name, version='0.1.0', packages=[package_name],
    data_files=[('share/ament_index/resource_index/packages', ['resource/' + package_name]),
                ('share/' + package_name, ['package.xml'])],
    install_requires=['setuptools'], zip_safe=True,
    maintainer='you', maintainer_email='you@example.com',
    description='ur_skills', license='MIT',
    entry_points={'console_scripts': ['pick_place = ur_skills.pick_place:main',
                                      'skill_server = ur_skills.skill_server:main']},
)
