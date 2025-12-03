from setuptools import setup

setup(name='vllm_looped_reasoner',
    version='0.1',
    packages=['vllm_looped_reasoner'],
    entry_points={
        'vllm.general_plugins':
        ["register_looped_reasoner = vllm_looped_reasoner.plugin:register"]
    })