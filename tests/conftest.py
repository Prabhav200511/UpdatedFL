import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bcpafl.blockchain import ROLE_BS, ROLE_RSU, ROLE_TA, BlockchainNetwork  # noqa: E402
from bcpafl.crypto.certificateless import KeyGenerationCenter  # noqa: E402
from bcpafl.identity import TrustedAuthority, VehicleWallet  # noqa: E402


class Infra:
    def __init__(self):
        self.kgc = KeyGenerationCenter()
        self.ta = TrustedAuthority(self.kgc)
        self.ta_kp = self.ta.issue_infrastructure_key("TA")
        self.bs_kp = self.ta.issue_infrastructure_key("BS")
        self.rsu_kp = self.ta.issue_infrastructure_key("RSU_0")
        self.rsu1_kp = self.ta.issue_infrastructure_key("RSU_1")
        self.chain = BlockchainNetwork(self.kgc.P_pub, {
            "TA": (ROLE_TA, self.ta_kp), "BS": (ROLE_BS, self.bs_kp),
            "RSU_0": (ROLE_RSU, self.rsu_kp), "RSU_1": (ROLE_RSU, self.rsu1_kp)})
        self.ta.attach_chain(self.chain, self.ta_kp)

    def vehicle(self, real_id, round_num=1, count=2, lifetime=3):
        self.ta.enroll(real_id)
        wallet = VehicleWallet(real_id, self.kgc.P_pub)
        self.ta.provision(wallet, round_num, count, lifetime)
        return wallet


@pytest.fixture
def infra():
    return Infra()
