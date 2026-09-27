Autoregressive version of ACMDM (Absolute Coordinates Motion Diffusion Model) inspired by JEPA. Simulates motion token by token in latent space which can be decoded to joint positions. Time to first token is about 0.1s and each frame takes about 0.06 seconds total to generate and decode so practically real-time motion generation. 

This is just the code; weights can be found [here](https://huggingface.co/abdilehia/ACJEPA). For setup instructions, see the original code repo which can be found in the acknowledgements. I mostly just did this out of curiosity as I thought the architecture would be suitable for it. Nothing about this is scientific and, for quality and prompt adherence, you are better off sticking to the diffusion version.

**Acknowledgements:**<br>
Meng, Z., Han, Z., Peng, X., Xie, Y., & Jiang, H. (2025). Absolute Coordinates Make Motion Generation Easy. arXiv Preprint arXiv:2505. 19377.<br>
Links to their paper [here](https://arxiv.org/pdf/2505.19377) and to their code implementation [here](https://github.com/neu-vi/ACMDM).
