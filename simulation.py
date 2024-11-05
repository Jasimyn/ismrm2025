import torch
import torch.nn as nn
import random
import torch.optim as optim
import torch.autograd.forward_ad as fwad
import math


def simulate_alpha(alpha, parameters, num_acqs):
    alpha = torch.squeeze(alpha).to(torch.device('cpu'))
    tf = 128
    if alpha.size()[0] < 10:
        alpha = torch.repeat_interleave(alpha, repeats=tf)
    # Specifying sequence parameters
    num_reps = 5
    # -number of TR's to simulate to achieve steady state signal
    esp = 5.74e-3
    # -time between flip angles in the echo trains

    # -Number of echoes in each echo train (turbo factor)
    ro_gap = 900e-3
    # -Gap between the acquistions in each TR
    time2rel = 0
    # -Relaxation time from last acquisition to end of TR

    b1_val = torch.tensor([1])
    # -B1+ for simulation
    inv_eff = torch.tensor([1])
    # -inversion efficiency for simulation
    etl = tf * esp
    # -length of each acquistion, given turbo factor and echo spacing

    # -sequence timings based on parameters defined above
    delT_M1_M2 = 109.7e-3
    # delT_M1_M2 = 85.58e-3
    delT_M0_M1 = ro_gap - etl - delT_M1_M2
    delT_M2_M3 = etl
    delT_M2_M6 = ro_gap
    delT_M4_M5 = 12.8e-3
    delT_M5_M6 = 100e-3 - 6.45e-3
    delT_M3_M4 = delT_M2_M6 - delT_M2_M3 - delT_M4_M5 - delT_M5_M6
    delT_M6_M7 = etl
    delT_M7_M8 = ro_gap - etl
    delT_M8_M9 = etl
    delT_M9_M10 = ro_gap - etl
    delT_M10_M11 = etl
    delT_M11_M12 = ro_gap - etl
    delT_M12_M13 = etl

    # -time between end of t2 prep pulse and first acquisition
    time_t2_prep_after = torch.tensor([9.7e-3])

    #####################################################################
    M0 = torch.tensor([1])
    Mz = torch.tensor([M0])

    Mxy_all = torch.zeros((num_acqs * tf, num_reps))

    for reps in range(num_reps):
        Mz = M0 - (M0 - Mz) * torch.exp(-delT_M0_M1 / parameters[1])
        Mz = Mz * (torch.sin(b1_val * torch.pi / 2) ** 2 * torch.exp(
            -(delT_M1_M2 - time_t2_prep_after) / parameters[0]) + \
                   torch.cos(b1_val * torch.pi / 2) ** 2 * torch.exp(
                    -(delT_M1_M2 - time_t2_prep_after) / parameters[1]))

        ech_ctr = 0
        acq_ctr = 0

        # ACQ1

        if (acq_ctr < num_acqs):
            for q in range(tf):
                if q == 0:
                    Mz = M0 - (M0 - Mz) * torch.exp(-time_t2_prep_after / parameters[1])
                else:
                    Mz = M0 - (M0 - Mz) * torch.exp(-esp / parameters[1])

                Mxy_all[ech_ctr, reps] = torch.sin(alpha[ech_ctr]) * Mz

                Mz = torch.cos(alpha[ech_ctr]) * Mz

                ech_ctr = ech_ctr + 1
            acq_ctr = acq_ctr + 1

        if (acq_ctr < num_acqs):
            Mz = M0 - (M0 - Mz) * torch.exp(-delT_M3_M4 / parameters[1])
            Mz = -Mz * inv_eff
            Mz = M0 - (M0 - Mz) * torch.exp(-delT_M5_M6 / parameters[1])

            # ACQ2
            for q in range(tf):
                if q > 0:
                    Mz = M0 - (M0 - Mz) * torch.exp(-esp / parameters[1])

                Mxy_all[ech_ctr, reps] = torch.sin(alpha[ech_ctr]) * Mz

                Mz = torch.cos(alpha[ech_ctr]) * Mz

                ech_ctr = ech_ctr + 1
            acq_ctr = acq_ctr + 1

        if (acq_ctr < num_acqs):
            Mz = M0 - (M0 - Mz) * torch.exp(-delT_M7_M8 / parameters[1])

            # ACQ3
            for q in range(tf):
                if q > 0:
                    Mz = M0 - (M0 - Mz) * torch.exp(-esp / parameters[1])

                Mxy_all[ech_ctr, reps] = torch.sin(alpha[ech_ctr]) * Mz

                Mz = torch.cos(alpha[ech_ctr]) * Mz

                ech_ctr = ech_ctr + 1
            acq_ctr = acq_ctr + 1

        if (acq_ctr < num_acqs):
            Mz = M0 - (M0 - Mz) * torch.exp(-delT_M9_M10 / parameters[1])

            # ACQ4
            for q in range(tf):
                if q > 0:
                    Mz = M0 - (M0 - Mz) * torch.exp(-esp / parameters[1])

                Mxy_all[ech_ctr, reps] = torch.sin(alpha[ech_ctr]) * Mz

                Mz = torch.cos(alpha[ech_ctr]) * Mz

                ech_ctr = ech_ctr + 1
            acq_ctr = acq_ctr + 1

        if (acq_ctr < num_acqs):
            Mz = M0 - (M0 - Mz) * torch.exp(-delT_M11_M12 / parameters[1])

            # ACQ5
            for q in range(tf):
                if q > 0:
                    Mz = M0 - (M0 - Mz) * torch.exp(-esp / parameters[1])

                Mxy_all[ech_ctr, reps] = torch.sin(alpha[ech_ctr]) * Mz

                Mz = torch.cos(alpha[ech_ctr]) * Mz

                ech_ctr = ech_ctr + 1

    result = Mxy_all[:, -1] * parameters[2]
    result = result.to(torch.device('cuda'))
    return result



def simulate_gap(gap, parameters, num_acqs):
    alpha = torch.ones(640).to(torch.device('cpu')) * 4 / 180 * math.pi
    ro_gap = torch.squeeze(gap).to(torch.device('cpu'))
    # Specifying sequence parameters
    num_reps = 5
    # -number of TR's to simulate to achieve steady state signal
    esp = 5.74e-3
    # -time between flip angles in the echo trains
    tf = 128
    # -Number of echoes in each echo train (turbo factor)
    # ro_gap = 900e-3
    # -Gap between the acquistions in each TR
    time2rel = 0
    # -Relaxation time from last acquisition to end of TR

    b1_val = torch.tensor([1])
    # -B1+ for simulation
    inv_eff = torch.tensor([1])
    # -inversion efficiency for simulation
    etl = tf * esp
    # -length of each acquistion, given turbo factor and echo spacing

    # -sequence timings based on parameters defined above
    delT_M1_M2 = 109.7e-3
    delT_M0_M1 = ro_gap[0] - etl - delT_M1_M2
    delT_M2_M3 = etl
    delT_M2_M6 = ro_gap[1]
    delT_M4_M5 = 12.8e-3
    delT_M5_M6 = 100e-3 - 6.45e-3
    delT_M3_M4 = delT_M2_M6 - delT_M2_M3 - delT_M4_M5 - delT_M5_M6
    delT_M6_M7 = etl
    delT_M7_M8 = ro_gap[2] - etl
    delT_M8_M9 = etl
    delT_M9_M10 = ro_gap[3] - etl
    delT_M10_M11 = etl
    delT_M11_M12 = ro_gap[4] - etl
    delT_M12_M13 = etl

    # -time between end of t2 prep pulse and first acquisition
    time_t2_prep_after = torch.tensor([9.7e-3])

    #####################################################################
    M0 = torch.tensor([1])
    Mz = torch.tensor([M0])

    Mxy_all = torch.zeros((num_acqs * tf, num_reps))

    for reps in range(num_reps):
        Mz = M0 - (M0 - Mz) * torch.exp(-delT_M0_M1 / parameters[1])
        Mz = Mz * (torch.sin(b1_val * torch.pi / 2) ** 2 * torch.exp(
            -(delT_M1_M2 - time_t2_prep_after) / parameters[0]) + \
                   torch.cos(b1_val * torch.pi / 2) ** 2 * torch.exp(
                    -(delT_M1_M2 - time_t2_prep_after) / parameters[1]))

        ech_ctr = 0
        acq_ctr = 0

        # ACQ1

        if (acq_ctr < num_acqs):
            for q in range(tf):
                if q == 0:
                    Mz = M0 - (M0 - Mz) * torch.exp(-time_t2_prep_after / parameters[1])
                else:
                    Mz = M0 - (M0 - Mz) * torch.exp(-esp / parameters[1])

                Mxy_all[ech_ctr, reps] = torch.sin(alpha[ech_ctr]) * Mz

                Mz = torch.cos(alpha[ech_ctr]) * Mz

                ech_ctr = ech_ctr + 1
            acq_ctr = acq_ctr + 1

        if (acq_ctr < num_acqs):
            Mz = M0 - (M0 - Mz) * torch.exp(-delT_M3_M4 / parameters[1])
            Mz = -Mz * inv_eff
            Mz = M0 - (M0 - Mz) * torch.exp(-delT_M5_M6 / parameters[1])

            # ACQ2
            for q in range(tf):
                if q > 0:
                    Mz = M0 - (M0 - Mz) * torch.exp(-esp / parameters[1])

                Mxy_all[ech_ctr, reps] = torch.sin(alpha[ech_ctr]) * Mz

                Mz = torch.cos(alpha[ech_ctr]) * Mz

                ech_ctr = ech_ctr + 1
            acq_ctr = acq_ctr + 1

        if (acq_ctr < num_acqs):
            Mz = M0 - (M0 - Mz) * torch.exp(-delT_M7_M8 / parameters[1])

            # ACQ3
            for q in range(tf):
                if q > 0:
                    Mz = M0 - (M0 - Mz) * torch.exp(-esp / parameters[1])

                Mxy_all[ech_ctr, reps] = torch.sin(alpha[ech_ctr]) * Mz

                Mz = torch.cos(alpha[ech_ctr]) * Mz

                ech_ctr = ech_ctr + 1
            acq_ctr = acq_ctr + 1

        if (acq_ctr < num_acqs):
            Mz = M0 - (M0 - Mz) * torch.exp(-delT_M9_M10 / parameters[1])

            # ACQ4
            for q in range(tf):
                if q > 0:
                    Mz = M0 - (M0 - Mz) * torch.exp(-esp / parameters[1])

                Mxy_all[ech_ctr, reps] = torch.sin(alpha[ech_ctr]) * Mz

                Mz = torch.cos(alpha[ech_ctr]) * Mz

                ech_ctr = ech_ctr + 1
            acq_ctr = acq_ctr + 1

        if (acq_ctr < num_acqs):
            Mz = M0 - (M0 - Mz) * torch.exp(-delT_M11_M12 / parameters[1])

            # ACQ5
            for q in range(tf):
                if q > 0:
                    Mz = M0 - (M0 - Mz) * torch.exp(-esp / parameters[1])

                Mxy_all[ech_ctr, reps] = torch.sin(alpha[ech_ctr]) * Mz

                Mz = torch.cos(alpha[ech_ctr]) * Mz

                ech_ctr = ech_ctr + 1

    result = Mxy_all[:, -1] * parameters[2]
    result = result.to(torch.device('cuda'))
    return result


def simulate_esp(esp, parameters, num_acqs):
    alpha = torch.ones(640).to(torch.device('cpu')) * 4 / 180 * math.pi
    esp = torch.squeeze(esp).to(torch.device('cpu'))
    # Specifying sequence parameters
    num_reps = 5
    # -number of TR's to simulate to achieve steady state signal
    # esp = 5.74e-3
    # -time between flip angles in the echo trains
    tf = 128
    # -Number of echoes in each echo train (turbo factor)
    ro_gap = 900e-3
    # -Gap between the acquistions in each TR
    time2rel = 0
    # -Relaxation time from last acquisition to end of TR

    b1_val = torch.tensor([1])
    # -B1+ for simulation
    inv_eff = torch.tensor([1])
    # -inversion efficiency for simulation
    etl = tf * esp
    # -length of each acquistion, given turbo factor and echo spacing

    # -sequence timings based on parameters defined above
    delT_M1_M2 = 109.7e-3
    delT_M0_M1 = ro_gap - etl - delT_M1_M2
    delT_M2_M3 = etl
    delT_M2_M6 = ro_gap
    delT_M4_M5 = 12.8e-3
    delT_M5_M6 = 100e-3 - 6.45e-3
    delT_M3_M4 = delT_M2_M6 - delT_M2_M3 - delT_M4_M5 - delT_M5_M6
    delT_M6_M7 = etl
    delT_M7_M8 = ro_gap - etl
    delT_M8_M9 = etl
    delT_M9_M10 = ro_gap - etl
    delT_M10_M11 = etl
    delT_M11_M12 = ro_gap - etl
    delT_M12_M13 = etl

    # -time between end of t2 prep pulse and first acquisition
    time_t2_prep_after = torch.tensor([9.7e-3])

    #####################################################################
    M0 = torch.tensor([1])
    Mz = torch.tensor([M0])

    Mxy_all = torch.zeros((num_acqs * tf, num_reps))

    for reps in range(num_reps):
        Mz = M0 - (M0 - Mz) * torch.exp(-delT_M0_M1 / parameters[1])
        Mz = Mz * (torch.sin(b1_val * torch.pi / 2) ** 2 * torch.exp(
            -(delT_M1_M2 - time_t2_prep_after) / parameters[0]) + \
                   torch.cos(b1_val * torch.pi / 2) ** 2 * torch.exp(
                    -(delT_M1_M2 - time_t2_prep_after) / parameters[1]))

        ech_ctr = 0
        acq_ctr = 0

        # ACQ1

        if (acq_ctr < num_acqs):
            for q in range(tf):
                if q == 0:
                    Mz = M0 - (M0 - Mz) * torch.exp(-time_t2_prep_after / parameters[1])
                else:
                    Mz = M0 - (M0 - Mz) * torch.exp(-esp / parameters[1])

                Mxy_all[ech_ctr, reps] = torch.sin(alpha[ech_ctr]) * Mz

                Mz = torch.cos(alpha[ech_ctr]) * Mz

                ech_ctr = ech_ctr + 1
            acq_ctr = acq_ctr + 1

        if (acq_ctr < num_acqs):
            Mz = M0 - (M0 - Mz) * torch.exp(-delT_M3_M4 / parameters[1])
            Mz = -Mz * inv_eff
            Mz = M0 - (M0 - Mz) * torch.exp(-delT_M5_M6 / parameters[1])

            # ACQ2
            for q in range(tf):
                if q > 0:
                    Mz = M0 - (M0 - Mz) * torch.exp(-esp / parameters[1])

                Mxy_all[ech_ctr, reps] = torch.sin(alpha[ech_ctr]) * Mz

                Mz = torch.cos(alpha[ech_ctr]) * Mz

                ech_ctr = ech_ctr + 1
            acq_ctr = acq_ctr + 1

        if (acq_ctr < num_acqs):
            Mz = M0 - (M0 - Mz) * torch.exp(-delT_M7_M8 / parameters[1])

            # ACQ3
            for q in range(tf):
                if q > 0:
                    Mz = M0 - (M0 - Mz) * torch.exp(-esp / parameters[1])

                Mxy_all[ech_ctr, reps] = torch.sin(alpha[ech_ctr]) * Mz

                Mz = torch.cos(alpha[ech_ctr]) * Mz

                ech_ctr = ech_ctr + 1
            acq_ctr = acq_ctr + 1

        if (acq_ctr < num_acqs):
            Mz = M0 - (M0 - Mz) * torch.exp(-delT_M9_M10 / parameters[1])

            # ACQ4
            for q in range(tf):
                if q > 0:
                    Mz = M0 - (M0 - Mz) * torch.exp(-esp / parameters[1])

                Mxy_all[ech_ctr, reps] = torch.sin(alpha[ech_ctr]) * Mz

                Mz = torch.cos(alpha[ech_ctr]) * Mz

                ech_ctr = ech_ctr + 1
            acq_ctr = acq_ctr + 1

        if (acq_ctr < num_acqs):
            Mz = M0 - (M0 - Mz) * torch.exp(-delT_M11_M12 / parameters[1])

            # ACQ5
            for q in range(tf):
                if q > 0:
                    Mz = M0 - (M0 - Mz) * torch.exp(-esp / parameters[1])

                Mxy_all[ech_ctr, reps] = torch.sin(alpha[ech_ctr]) * Mz

                Mz = torch.cos(alpha[ech_ctr]) * Mz

                ech_ctr = ech_ctr + 1

    result = Mxy_all[:, -1] * parameters[2]
    result = result.to(torch.device('cuda'))
    return result


def simulate_joint(fa, gap, prep, parameters, num_acqs):
    alpha = torch.squeeze(fa).to(torch.device('cpu'))
    ro_gap = torch.squeeze(gap).to(torch.device('cpu'))
    prep = torch.squeeze(prep).to(torch.device('cpu'))
    num_reps = 5
    # -number of TR's to simulate to achieve steady state signal
    esp = 5.74e-3
    # -time between flip angles in the echo trains
    tf = 128
    # -Number of echoes in each echo train (turbo factor)
    # ro_gap = 900e-3
    # -Gap between the acquistions in each TR
    time2rel = 0
    # -Relaxation time from last acquisition to end of TR

    b1_val = torch.tensor([1])
    # -B1+ for simulation
    inv_eff = torch.tensor([1])
    # -inversion efficiency for simulation
    etl = tf * esp
    # -length of each acquistion, given turbo factor and echo spacing

    # -sequence timings based on parameters defined above
    delT_M1_M2 = prep
    delT_M0_M1 = ro_gap[0] - etl - delT_M1_M2
    delT_M2_M3 = etl
    delT_M2_M6 = ro_gap[1]
    delT_M4_M5 = 12.8e-3
    delT_M5_M6 = 100e-3 - 6.45e-3
    delT_M3_M4 = delT_M2_M6 - delT_M2_M3 - delT_M4_M5 - delT_M5_M6
    delT_M6_M7 = etl
    delT_M7_M8 = ro_gap[2] - etl
    delT_M8_M9 = etl
    delT_M9_M10 = ro_gap[3] - etl
    delT_M10_M11 = etl
    delT_M11_M12 = ro_gap[4] - etl
    delT_M12_M13 = etl

    # -time between end of t2 prep pulse and first acquisition
    time_t2_prep_after = torch.tensor([9.7e-3])

    #####################################################################
    M0 = torch.tensor([1])
    Mz = torch.tensor([M0])

    Mxy_all = torch.zeros((num_acqs * tf, num_reps))

    for reps in range(num_reps):
        Mz = M0 - (M0 - Mz) * torch.exp(-delT_M0_M1 / parameters[1])
        Mz = Mz * (torch.sin(b1_val * torch.pi / 2) ** 2 * torch.exp(
            -(delT_M1_M2 - time_t2_prep_after) / parameters[0]) + \
                   torch.cos(b1_val * torch.pi / 2) ** 2 * torch.exp(
                    -(delT_M1_M2 - time_t2_prep_after) / parameters[1]))

        ech_ctr = 0
        acq_ctr = 0

        # ACQ1

        if (acq_ctr < num_acqs):
            for q in range(tf):
                if q == 0:
                    Mz = M0 - (M0 - Mz) * torch.exp(-time_t2_prep_after / parameters[1])
                else:
                    Mz = M0 - (M0 - Mz) * torch.exp(-esp / parameters[1])

                Mxy_all[ech_ctr, reps] = torch.sin(alpha[ech_ctr]) * Mz

                Mz = torch.cos(alpha[ech_ctr]) * Mz

                ech_ctr = ech_ctr + 1
            acq_ctr = acq_ctr + 1

        if (acq_ctr < num_acqs):
            Mz = M0 - (M0 - Mz) * torch.exp(-delT_M3_M4 / parameters[1])
            Mz = -Mz * inv_eff
            Mz = M0 - (M0 - Mz) * torch.exp(-delT_M5_M6 / parameters[1])

            # ACQ2
            for q in range(tf):
                if q > 0:
                    Mz = M0 - (M0 - Mz) * torch.exp(-esp / parameters[1])

                Mxy_all[ech_ctr, reps] = torch.sin(alpha[ech_ctr]) * Mz

                Mz = torch.cos(alpha[ech_ctr]) * Mz

                ech_ctr = ech_ctr + 1
            acq_ctr = acq_ctr + 1

        if (acq_ctr < num_acqs):
            Mz = M0 - (M0 - Mz) * torch.exp(-delT_M7_M8 / parameters[1])

            # ACQ3
            for q in range(tf):
                if q > 0:
                    Mz = M0 - (M0 - Mz) * torch.exp(-esp / parameters[1])

                Mxy_all[ech_ctr, reps] = torch.sin(alpha[ech_ctr]) * Mz

                Mz = torch.cos(alpha[ech_ctr]) * Mz

                ech_ctr = ech_ctr + 1
            acq_ctr = acq_ctr + 1

        if (acq_ctr < num_acqs):
            Mz = M0 - (M0 - Mz) * torch.exp(-delT_M9_M10 / parameters[1])

            # ACQ4
            for q in range(tf):
                if q > 0:
                    Mz = M0 - (M0 - Mz) * torch.exp(-esp / parameters[1])

                Mxy_all[ech_ctr, reps] = torch.sin(alpha[ech_ctr]) * Mz

                Mz = torch.cos(alpha[ech_ctr]) * Mz

                ech_ctr = ech_ctr + 1
            acq_ctr = acq_ctr + 1

        if (acq_ctr < num_acqs):
            Mz = M0 - (M0 - Mz) * torch.exp(-delT_M11_M12 / parameters[1])

            # ACQ5
            for q in range(tf):
                if q > 0:
                    Mz = M0 - (M0 - Mz) * torch.exp(-esp / parameters[1])

                Mxy_all[ech_ctr, reps] = torch.sin(alpha[ech_ctr]) * Mz

                Mz = torch.cos(alpha[ech_ctr]) * Mz

                ech_ctr = ech_ctr + 1

    result = Mxy_all[:, -1] * parameters[2]
    result = result.to(torch.device('cuda'))
    return result


def simulate_alpha_basedGP(alpha, parameters, num_acqs):
    alpha = torch.squeeze(alpha).to(torch.device('cpu'))
    ro_gap = torch.squeeze(torch.tensor([1000e-3, 932.6e-3, 750e-3, 1000e-3, 1000e-3])).to(torch.device('cpu'))
    # Specifying sequence parameters
    num_reps = 5
    # -number of TR's to simulate to achieve steady state signal
    esp = 5.74e-3
    # -time between flip angles in the echo trains
    tf = 128
    # -Number of echoes in each echo train (turbo factor)
    # ro_gap = 900e-3
    # -Gap between the acquistions in each TR
    time2rel = 0
    # -Relaxation time from last acquisition to end of TR

    b1_val = torch.tensor([1])
    # -B1+ for simulation
    inv_eff = torch.tensor([1])
    # -inversion efficiency for simulation
    etl = tf * esp
    # -length of each acquistion, given turbo factor and echo spacing

    # -sequence timings based on parameters defined above
    # delT_M1_M2 = 109.7e-3
    delT_M1_M2 = 85.57e-3
    delT_M0_M1 = ro_gap[0] - etl - delT_M1_M2
    delT_M2_M3 = etl
    delT_M2_M6 = ro_gap[1]
    delT_M4_M5 = 12.8e-3
    delT_M5_M6 = 100e-3 - 6.45e-3
    delT_M3_M4 = delT_M2_M6 - delT_M2_M3 - delT_M4_M5 - delT_M5_M6
    delT_M6_M7 = etl
    delT_M7_M8 = ro_gap[2] - etl
    delT_M8_M9 = etl
    delT_M9_M10 = ro_gap[3] - etl
    delT_M10_M11 = etl
    delT_M11_M12 = ro_gap[4] - etl
    delT_M12_M13 = etl

    # -time between end of t2 prep pulse and first acquisition
    time_t2_prep_after = torch.tensor([9.7e-3])

    #####################################################################
    M0 = torch.tensor([1])
    Mz = torch.tensor([M0])

    Mxy_all = torch.zeros((num_acqs * tf, num_reps))

    for reps in range(num_reps):
        Mz = M0 - (M0 - Mz) * torch.exp(-delT_M0_M1 / parameters[1])
        Mz = Mz * (torch.sin(b1_val * torch.pi / 2) ** 2 * torch.exp(
            -(delT_M1_M2 - time_t2_prep_after) / parameters[0]) + \
                   torch.cos(b1_val * torch.pi / 2) ** 2 * torch.exp(
                    -(delT_M1_M2 - time_t2_prep_after) / parameters[1]))

        ech_ctr = 0
        acq_ctr = 0

        # ACQ1

        if (acq_ctr < num_acqs):
            for q in range(tf):
                if q == 0:
                    Mz = M0 - (M0 - Mz) * torch.exp(-time_t2_prep_after / parameters[1])
                else:
                    Mz = M0 - (M0 - Mz) * torch.exp(-esp / parameters[1])

                Mxy_all[ech_ctr, reps] = torch.sin(alpha[ech_ctr]) * Mz

                Mz = torch.cos(alpha[ech_ctr]) * Mz

                ech_ctr = ech_ctr + 1
            acq_ctr = acq_ctr + 1

        if (acq_ctr < num_acqs):
            Mz = M0 - (M0 - Mz) * torch.exp(-delT_M3_M4 / parameters[1])
            Mz = -Mz * inv_eff
            Mz = M0 - (M0 - Mz) * torch.exp(-delT_M5_M6 / parameters[1])

            # ACQ2
            for q in range(tf):
                if q > 0:
                    Mz = M0 - (M0 - Mz) * torch.exp(-esp / parameters[1])

                Mxy_all[ech_ctr, reps] = torch.sin(alpha[ech_ctr]) * Mz

                Mz = torch.cos(alpha[ech_ctr]) * Mz

                ech_ctr = ech_ctr + 1
            acq_ctr = acq_ctr + 1

        if (acq_ctr < num_acqs):
            Mz = M0 - (M0 - Mz) * torch.exp(-delT_M7_M8 / parameters[1])

            # ACQ3
            for q in range(tf):
                if q > 0:
                    Mz = M0 - (M0 - Mz) * torch.exp(-esp / parameters[1])

                Mxy_all[ech_ctr, reps] = torch.sin(alpha[ech_ctr]) * Mz

                Mz = torch.cos(alpha[ech_ctr]) * Mz

                ech_ctr = ech_ctr + 1
            acq_ctr = acq_ctr + 1

        if (acq_ctr < num_acqs):
            Mz = M0 - (M0 - Mz) * torch.exp(-delT_M9_M10 / parameters[1])

            # ACQ4
            for q in range(tf):
                if q > 0:
                    Mz = M0 - (M0 - Mz) * torch.exp(-esp / parameters[1])

                Mxy_all[ech_ctr, reps] = torch.sin(alpha[ech_ctr]) * Mz

                Mz = torch.cos(alpha[ech_ctr]) * Mz

                ech_ctr = ech_ctr + 1
            acq_ctr = acq_ctr + 1

        if (acq_ctr < num_acqs):
            Mz = M0 - (M0 - Mz) * torch.exp(-delT_M11_M12 / parameters[1])

            # ACQ5
            for q in range(tf):
                if q > 0:
                    Mz = M0 - (M0 - Mz) * torch.exp(-esp / parameters[1])

                Mxy_all[ech_ctr, reps] = torch.sin(alpha[ech_ctr]) * Mz

                Mz = torch.cos(alpha[ech_ctr]) * Mz

                ech_ctr = ech_ctr + 1

    result = Mxy_all[:, -1] * parameters[2]
    result = result.to(torch.device('cuda'))
    return result
