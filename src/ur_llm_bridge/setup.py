from setuptools import setup
package_name = 'ur_llm_bridge'
setup(
    name=package_name, version='0.1.0', packages=[package_name],
    data_files=[('share/ament_index/resource_index/packages', ['resource/' + package_name]),
                ('share/' + package_name, ['package.xml'])],
    install_requires=['setuptools'], zip_safe=True,
    maintainer='you', maintainer_email='you@example.com',
    description='Local LLM (Ollama) to ROS 2 skill bridge', license='MIT',
    entry_points={'console_scripts': [
        'llm_bridge = ur_llm_bridge.llm_bridge:main',
        'llm_benchmark = ur_llm_bridge.benchmark:main',
        'reset_scene = ur_llm_bridge.benchmark:reset_main']},
)
