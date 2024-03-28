

def loss_function_vae(x_obs, z_obs, q_x, q_z_given_x, p_z, p_x_given_z,
                      samp_p=False, binary_x=False, beta=1.0):
    # vae loss
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # cross entropy (binary or continuous)
    CE = -p_x_given_z.log_prob(x_obs)

    # KL divergence
    # KLD = torch.distributions.kl.kl_divergence(q_z_given_x, p_z).sum(-1)
    KLD = q_z_given_x.log_prob(z_obs) - p_z.log_prob(z_obs)

    loss = (CE + beta * KLD).mean()

    return loss

def loss_function_mim(x_obs, z_obs, q_x, q_z_given_x, p_z, p_x_given_z,
                      samp_p=False, binary_x=False, beta=1.0):

    # MIM loss
    log_prob_x_given_z = p_x_given_z.log_prob(x_obs)
    log_prob_x = q_x.log_prob(x_obs)

    log_prob_z = p_z.log_prob(z_obs)
    log_prob_z_given_x = q_z_given_x.log_prob(z_obs)

    loss = -0.5 * (log_prob_x_given_z + log_prob_z + beta * (log_prob_z_given_x + log_prob_x))

    # REINFORCE
    if binary_x and samp_p:
        loss = loss + loss.detach() * log_prob_x_given_z - (loss * log_prob_x_given_z).detach()

    loss = loss.mean()

    return loss

def loss_function_amim(self, x, beta=1., average=False):
        # pass through VAE
        x_mean, x_logvar, z1_q, z1_q_mean, z1_q_logvar, z2_q, z2_q_mean, z2_q_logvar, z1_p_mean, z1_p_logvar = self.forward(
            x)

        # p(x|z)p(z)
        if self.args.input_type == 'binary':
            log_p_x_given_z = log_Bernoulli(x, x_mean, dim=1)
        elif self.args.input_type == 'gray' or self.args.input_type == 'continuous':
            log_p_x_given_z = -log_Logistic_256(x, x_mean, x_logvar, dim=1)
        else:
            raise Exception('Wrong input type!')

        log_p_z1 = log_Normal_diag(z1_q, z1_p_mean, z1_p_logvar, dim=1)
        log_p_z2 = self.log_p_z2(z2_q)
        log_p_z = log_p_z1 + beta * log_p_z2

        # q(z|x)q(x)
        log_q_z1 = log_Normal_diag(z1_q, z1_q_mean, z1_q_logvar, dim=1)
        log_q_z2 = log_Normal_diag(z2_q, z2_q_mean, z2_q_logvar, dim=1)
        log_q_z_given_x = log_q_z1 + log_q_z2

        if self.args.q_x_prior == "marginal":
            # q(x) is marginal of p(x, z)
            log_q_x = log_p_x_given_z + log_p_z - log_q_z_given_x
        elif self.args.q_x_prior == "vampprior":
            # q(x) is vamprior of p(x|u)
            log_q_x = self.log_q_x_vampprior(x)

        RE = log_p_x_given_z
        KL = -(log_p_z1 + log_p_z2 - log_q_z1 - log_q_z2)

        # MIM loss
        loss = -0.5 * (log_p_x_given_z + log_p_z + beta * (log_q_z_given_x + log_q_x))

        if average:
            loss = torch.mean(loss)
            RE = torch.mean(RE)
            KL = torch.mean(KL)